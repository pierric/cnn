{-# LANGUAGE BangPatterns, TypeFamilies, TypeOperators, FlexibleInstances, FlexibleContexts #-}
module CNN where
import Numeric.LinearAlgebra hiding (R)
import Numeric.LinearAlgebra.Devel
import qualified Data.Vector as V
import qualified Data.Vector.Mutable as VM
import qualified Data.Vector.Storable as SV
import qualified Data.Vector.Storable.Mutable as SVM
import System.Random.MWC
import System.Random.MWC.Distributions
import Control.Monad.ST
import Control.Monad (liftM2, forM_, when)
import Control.Exception (assert)
import GHC.Float
import Utils
import Debug.Trace
import Text.PrettyPrint.Free hiding (flatten, (<>))
import Text.Printf (printf)
import Data.STRef
import Data.List (maximumBy)
import System.IO.Unsafe ( unsafePerformIO )

type R = Float

class Component a where
  type Inp a
  type Out a
  -- the trace start with the input, following
  -- by weighted-sum value and activated value
  -- of each inner level.
  data Trace a
  -- Forward propagation
  -- input: layer, input value
  -- output: a trace
  forwardT :: a -> Inp a -> Trace a
  -- Forward propagation
  -- input: layer, input value
  -- output: the final activated value.
  forward  :: a -> Inp a -> Out a
  forward a = output . forwardT a
  -- extract the final activated value from the trace
  output   :: Trace a -> Out a
  -- input:  old layer, a trace, out-error, learning rate
  -- output: new layer, in-error
  learn :: a -> Trace a -> Out a -> R -> (a, Inp a)

-- Densely connected layer
-- input:   vector of size m
-- output:  vector of size n
-- weights: matrix of size m x n
-- biases:  vector of size n
data DLayer = DLayer {
  dweights :: !(Matrix R), dbiases :: !(Vector R),
  activate :: R -> R, activate' :: R -> R
}

instance Component DLayer where
    type Inp DLayer = Vector R
    type Out DLayer = Vector R
    -- trace is (input, weighted-sum, output)
    newtype Trace DLayer = DTrace (Vector R, Vector R, Vector R)
    forwardT (DLayer{dweights=w,dbiases=b,activate=af}) !inp =
        let !bv = (inp <# w) `add` b
            !av = cmap af bv
        in DTrace (inp,bv,av)
    output (DTrace (_,_,!a)) = a
    learn l (DTrace (!iv,!bv,_)) !odelta rate =
        let DLayer{dweights=w,dbiases=b,activate'=af'} = l
            -- back-propagated error before activation
            udelta = odelta `hadamard` cmap af' bv
            -- back-propagated error at input
            !idelta = w #> udelta
            -- update to biases
            ---- for what reason, could this expression
            ---- entails a huge space leak?
            !d = b `add` scale (negate rate) udelta
            -- update to weights
            ---- for what reason, could this expression: w `add` (iv `outer` d)
            ---- entails a huge space leak? especially, neither 'seq' nor
            ---- 'deepseq' helps a bit. The only workaround is to expand the
            ---- add function, and call SV.force on the result vector, which
            ---- explcitly copy and drop reference to orignal computed result.
            !m = let (r,c) = size w
                     dat1 = flatten (tr' w)
                     dat2 = flatten (tr' (iv `outer` d))
                 in matrixFromVector ColumnMajor r c $ SV.force $ dat1 `add` dat2
        in (l{dweights=m, dbiases=d}, idelta)

-- convolutional layer
-- input:  channels of 2D floats, of the same size (a x b), # of input channels:  m
-- output: channels of 2D floats, of the same size (c x d), # of output channels: n
--         where c = a + 2*padding + 1 - s
--               d = b + 2*padding + 1 - t
-- feature:  matrix of (s x t), # of features: m x n
-- padding:  number of 0s padded at each side of channel
-- biases:   bias for each output, # of biases: n
data CLayer = CLayer {
  cfilters :: !(V.Vector (V.Vector (Matrix R))), -- feature matrix indexed majorly by each input
  cbiases  :: !(V.Vector R),
  cpadding :: !Int
}

instance Component CLayer where
    type Inp CLayer = V.Vector (Matrix R)
    type Out CLayer = V.Vector (Matrix R)
    -- trace is (input, convoluted output)
    newtype Trace CLayer = CTrace (Inp CLayer, V.Vector (Matrix R))
    forwardT (CLayer{cfilters=fs, cbiases=bs, cpadding=p}) !inp =
        let !ov = parallel $ V.zipWith feature
                               (tr fs) -- feature matrix indexed majorly by each output
                               bs      -- biases by each output
        in CTrace (inp,ov)
      where
        !osize = let (x,y) = size (V.head inp)
                     (u,v) = size (V.head $ V.head fs)
                 in (x+2*p-u+1, y+2*p-v+1)
        -- transpose the features matrix
        tr :: V.Vector (V.Vector a) -> V.Vector (V.Vector a)
        tr uv = let n   = V.length (V.head uv)
                    !vu = V.map (\i -> V.map (V.! i) uv) $ V.enumFromN 0 n
                in vu
        feature :: V.Vector (Matrix R) -> R -> Matrix R
        feature f b = V.foldl1' add (V.zipWith (layerCorr2 p) f inp) `add` konst b osize
    output (CTrace (_,a)) = a
    learn l (CTrace (!iv,!av)) !odelta rate =
      let CLayer{cfilters=fs, cbiases=bs, cpadding=p} = l
          -- update to the feature matrix
          m :: V.Vector (V.Vector (Matrix R))
          !m = parallel $ V.zipWith (\flts chn ->
                            -- chn:  a single input channel
                            -- flts: all features used for chn
                            V.zipWith (\f d ->
                              let upd = scale (negate rate) (layerCorr2 p chn d)
                              in f `add` upd
                            ) flts odelta
                          ) fs iv
          -- update to the biases
          b :: V.Vector R
          !b = V.zipWith (\b d -> b + (negate rate) * sumElements d) bs odelta
          -- back-propagated error at input
          idelta :: V.Vector (Matrix R)
          !idelta = V.map (\f -> V.foldl1' add $ V.zipWith (layerConv2 p) f odelta) fs
      in --trace ("CL:" ++ show odelta)
         (l{ cfilters= m, cbiases = b}, idelta)

-- max pooling layer
-- input:  channels of 2D floats, of the same size (a x b), # of input channels:  m
--         assuming that a and b are both multiple of stride
-- output: channels of 2D floats, of the same size (c x d), # of output channels: m
--         where c = a / stride
--               d = b / stride

data MaxPoolLayer = MaxPoolLayer Int

instance Component MaxPoolLayer where
  type Inp MaxPoolLayer = V.Vector (Matrix R)
  type Out MaxPoolLayer = V.Vector (Matrix R)
  -- trace is (dimension of pools, index of max in each pool, pooled matrix)
  -- for each channel.
  newtype Trace MaxPoolLayer = PTrace (V.Vector (IndexOf Matrix, Vector Int, Matrix R))
  -- forward is to divide the input matrix in stride x stride sub matrices,
  -- and then find the max element in each sub matrices.
  forwardT (MaxPoolLayer stride) !inp = PTrace $ parallel $ V.map mk inp
    where
      mk inp = let (!i,!v) = pool stride inp in (size v, i, v)
  output (PTrace a) = V.map (\(_,_,!o) ->o) a
  -- use the saved index-of-max in each pool to propagate the error.
  learn l@(MaxPoolLayer stride) (PTrace t) odelta _ =
      let !idelta = V.zipWith gen t odelta in (l, idelta)
    where
      gen (!si,!iv,_) od = assert (si == size od) $ unpool stride iv od

-- Relu activator for single channel input
-- the input can be either a 1D vector or 2D matrix
data ReluLayerS a = ReluLayerS
instance (Container c R) => Component (ReluLayerS (c R)) where
    type Inp (ReluLayerS (c R)) = c R
    type Out (ReluLayerS (c R)) = c R
    newtype Trace (ReluLayerS (c R)) = ReluTrace (c R, c R)
    forwardT _ !inp = ReluTrace (inp, cmap relu inp)
    output (ReluTrace (_,!a)) = a
    learn a (ReluTrace (!iv,_)) !odelta _ = (a, odelta `hadamard` cmap relu' iv)

-- Relu activator for multiple channels input
data ReluLayerM a = ReluLayerM
instance (Container c R) => Component (ReluLayerM (c R)) where
    type Inp (ReluLayerM (c R)) = V.Vector (c R)
    type Out (ReluLayerM (c R)) = V.Vector (c R)
    newtype Trace (ReluLayerM (c R)) = ReluTraceM (V.Vector (Trace (ReluLayerS (c R))))
    forwardT _ !inp = ReluTraceM $ V.map (forwardT ReluLayerS) inp
    output (ReluTraceM a) = V.map output a
    learn a (ReluTraceM ts) !odelta r = (a, V.zipWith (\t d -> snd $ learn ReluLayerS t d r) ts odelta)

-- Reshape from channels of matrix to a single vector
-- input:  m channels of 2D matrices
--         assuming that all matrices are of the same size a x b
-- output: 1D vector of the concatenation of all input channels
--         its size: m x a x b
data ReshapeLayer = ReshapeLayer
instance Component ReshapeLayer where
  type Inp ReshapeLayer = V.Vector (Matrix R)
  type Out ReshapeLayer = Vector R
  -- trace keeps information of (m, axb, b, output)
  newtype Trace ReshapeLayer = ReshapeTrace (Int, Int, Int, Vector R)
  forwardT _ !inp =
    let !b = V.length inp
        (!r,!c) = size (V.head inp)
        !o = V.foldr' (\x y -> flatten x SV.++ y) SV.empty inp
    in ReshapeTrace (b, r*c, c, o)
  output (ReshapeTrace (_,_,_,a)) = a
  learn a (ReshapeTrace (b,n,c,_)) !odelta _ =
    let !idelta = V.fromList $ map (reshape c) $ takesV (replicate b n) odelta
    in (a, idelta)

-- stacking two components a and b
-- the output of a should matches the input of b
data a :+> b = !a :+> !b
instance (Component a, Component b, Out a ~ Inp b) => Component (a :+> b) where
    type Inp (a :+> b) = Inp a
    type Out (a :+> b) = Out b
    newtype Trace (a :+> b) = TTrace (Trace b, Trace a)
    forwardT (a:+>b) !i =
        let !tra = forwardT a i
            !trb = forwardT b (output tra)
        in TTrace (trb, tra)
    output (TTrace !a) = output (fst a)
    learn (a:+>b) (TTrace (!trb,!tra)) !odelta rate =
        let (b', !odelta') = learn b trb odelta  rate
            (a', !idelta ) = learn a tra odelta' rate
        in (a':+>b', idelta)

infixr 0 ++>
(++>) :: (Monad m, Component a, Component b, Component (a :+> b))
      => m a -> m b -> m (a :+> b)
(++>) = liftM2 (:+>)

newDLayer :: (Int, Int)         -- number of input channels, number of neurons (output channels)
          -> (R->R, R->R)       -- activate function and its derivative
          -> IO DLayer          -- new layer
newDLayer sz@(_,n) (af, af') =
    withSystemRandom . asGenIO $ \gen -> do
        -- we build the weights in column major because in the back-propagation
        -- algo, the computed update to weights is in column major. So it is
        -- good for performance to keep the matrix always in column major.
        w <- buildMatrix (normal 0 0.01 gen) ColumnMajor sz
        b <- return $ konst 1 n
        return $ DLayer w b af af'

newCLayer :: Int                -- number of input channels
          -> Int                -- number of output channels
          -> Int                -- size of each feature
          -> Int                -- size of padding
          -> IO CLayer          -- new layer
newCLayer inpsize outsize sfilter npadding =
  withSystemRandom . asGenIO $ \gen -> do
      fs <- V.replicateM inpsize $ V.replicateM outsize $
              buildMatrix (truncNormal 0 0.1 gen) RowMajor (sfilter, sfilter)
      bs <- return $ V.replicate outsize 0.1
      return $ CLayer fs bs npadding
  where
    truncNormal m s g = do
      x <- standard g
      if x >= 2.0 || x <= -2.0
        then truncNormal m s g
        else return $! m + s * x

buildMatrix g order (nr, nc) = do
  vals <- SV.replicateM (nr*nc) (double2Float <$> g)
  return $ matrixFromVector order nr nc vals

maxPool :: Int -> IO MaxPoolLayer
maxPool = return . MaxPoolLayer

reshapeL :: IO ReshapeLayer
reshapeL = return ReshapeLayer

reluMulti :: IO (ReluLayerM (Matrix R))
reluMulti = return ReluLayerM

learnStep :: Component n
    => (Out n -> Out n -> Out n)  -- derivative of the cost function
    -> R                          -- learning rate
    -> n                          -- neuron network
    -> (Inp n, Out n)             -- input and expect output
    -> n                          -- updated network
learnStep cost rate n (i,o) =
    let tr = forwardT n i
    in fst $ learn n tr (cost (output tr) o) rate

layerCorr2 :: Int -> Matrix R -> Matrix R -> Matrix R
layerCorr2 p k m = c_corr2d_s k padded
  where
    padded = zeroPadded p m
    (w,_)  = size k

layerConv2 :: Int -> Matrix R -> Matrix R -> Matrix R
layerConv2 p k m = c_conv2d_s k padded
  where
    padded = zeroPadded p m
    (w,_)  = size k

relu, relu' :: R-> R
relu = max 0
relu' x | x < 0     = 0
        | otherwise = 1

cost' a y | y == 1 && a >= y = 0
          | otherwise        = a - y

-- max pool, picking out the maximum element
-- in each stride x stride sub-matrices.
-- assuming that the original matrix row and column size are
-- both multiple of stride
pool :: Int -> Matrix Float -> (Vector Int, Matrix Float)
pool 1 mat = let (r,c) = size mat in (SV.replicate (r*c) 0, mat)
-- pool 2 mat | orderOf mat == RowMajor = c_max_pool2_f mat
pool stride mat = runST $ do
  ori <- unsafeThawMatrix mat
  mxv <- newUndefinedMatrix RowMajor r' c'
  mxi <- newUndefinedVector (r'*c')
  forM_ [0..r'-1] $ \i -> do
    forM_ [0..c'-1] $ \j -> do
      (n,v) <- unsafeMaxIndEle ori (i*stride) (j*stride) stride stride
      unsafeWriteVector mxi (i*c'+j) n
      unsafeWriteMatrix mxv i j v
  a <- unsafeFreezeVector mxi
  b <- unsafeFreezeMatrix mxv
  return (a,b)
  where
    (r,c) = size mat
    r'    = r `div` stride
    c'    = c `div` stride
    unsafeMaxIndEle mm x y r c = do
      mp <- newSTRef 0
      mv <- newSTRef (-10000.0)
      forM_ [0..r-1] $ \ i -> do
        forM_ [0..c-1] $ \ j -> do
          v1 <- unsafeReadMatrix mm (x+i) (y+j)
          v0 <- readSTRef mv
          when (v1 > v0) $ do
            writeSTRef mv v1
            writeSTRef mp (i*2+j)
      p <- readSTRef mp
      v <- readSTRef mv
      return (p, v)

-- the reverse of max pool.
-- assuming idx and mat are of the same size
unpool :: Int -> Vector Int -> Matrix Float -> Matrix Float
unpool stride idx mat = runSTMatrix $ do
  mat' <- newMatrix' 0 r' c'
  forM_ [0..r-1] $ \i -> do
    forM_ [0..c-1] $ \j -> do
      let pos     = idx SV.! (i*c+j)
      let (oi,oj) = pos `divMod` 2
      let val     = mat `atIndex` (i,j)
      unsafeWriteMatrix mat' (i*stride+oi) (j*stride+oj) val
  return mat'
  where
    (r,c) = size mat
    (r',c') = (r*stride, c*stride)

-- a slightly faster way to pading the matrix
-- camparing to fromBlocks provided by hmatrix.
zeroPadded :: Int -> Matrix Float -> Matrix Float
zeroPadded p mat = runSTMatrix $ do
  mat' <- newMatrix' 0 r' c'
  setMatrix mat' p p mat
  return mat'
  where
    (r,c) = size mat
    (r',c') = (r+2*p,c+2*p)

-- a slightly faster version of newMatrix, which based
-- directly on lower level Vector.Storable creation.
newMatrix' :: SVM.Storable t => t -> Int -> Int -> ST s (STMatrix s t)
newMatrix' v r c = do
  vec <- SVM.replicate (r*c) v
  vec <- SV.unsafeFreeze vec
  unsafeThawMatrix $ reshape c vec

--
-- for debugging purpose
--
instance Pretty (Matrix R) where
  pretty mat = pretty (dispf 2 $ double mat)
instance Pretty (Vector R) where
  pretty vec = encloseSep langle rangle comma $ map (text . printf "%.04f") $ SV.toList vec
instance Pretty CLayer where
  pretty (CLayer f b p) =
    let tm = tupled $ V.map pretty f
        tb = text "biased by " <+> (tupled $ V.map pretty b)
        tp = text "padded by " <+> pretty p
    in vsep [text "==>CL", tm, tb, tp]
instance Pretty DLayer where
  pretty (DLayer w b _ _) =
    let tm = pretty w
        tb = text "biased by " <+> pretty b
    in vsep [text "==>DL", tm, tb]
instance Pretty ReshapeLayer where
  pretty (ReshapeLayer) =
    text "==>RL"
instance Pretty MaxPoolLayer where
  pretty (MaxPoolLayer n) =
    hcat [text "==>ML(", pretty n, text ")"]
instance Pretty (ReluLayerM a) where
  pretty (ReluLayerM) =
    text "==>AL"
instance (Pretty a, Pretty b) => Pretty (a :+> b) where
  pretty (a:+>b) =
    let sep = hcat (replicate 40 $ char '~')
    in vsep [pretty a, sep, pretty b]
instance Pretty a => Pretty (V.Vector a) where
  pretty vec = list (V.map pretty vec)
instance Show CLayer where
  show = show . pretty
instance Show DLayer where
  show = show . pretty
instance (Pretty a, Pretty b) => Show (a :+> b) where
  show = show . pretty
