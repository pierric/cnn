{-# LANGUAGE BangPatterns, TypeFamilies, TypeOperators, FlexibleInstances, FlexibleContexts #-}
module CNN where
import Numeric.LinearAlgebra hiding (R)
import Numeric.LinearAlgebra.Devel
import qualified Data.Vector as V
import qualified Data.Vector.Storable as SV
import System.Random.MWC
import System.Random.MWC.Distributions
import Control.Monad.ST
import Control.Monad (liftM2)
import Control.Exception (assert)
import GHC.Float
import Utils

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

data DLayer = DLayer {
  dweights :: !(Matrix R), dbiases :: !(Vector R),
  activate :: R -> R, activate' :: R -> R
}

instance Component DLayer where
    type Inp DLayer = Vector R
    type Out DLayer = Vector R
    -- trace is (input, weighted-sum, activated output)
    newtype Trace DLayer = DTrace (Vector R, Vector R, Vector R)
    forwardT (DLayer{dweights=w,dbiases=b,activate=af}) !inp =
        let !bv = (inp <# w) `add` b
            !av = cmap af bv
        in DTrace (inp,bv,av)
    output (DTrace (_,_,!a)) = a
    learn l (DTrace (!iv,!bv,_)) !odelta rate =
        let DLayer{dweights=w,dbiases=b,activate'=af'} = l
            udelta = odelta `hadamard` cmap af' bv
            !d = scale (negate rate) udelta
            !m = iv `outer` d
            !idelta = w #> udelta
        in (l{dweights=w `add` m, dbiases=b `add` d}, idelta)

data CLayer = CLayer {
  cfilters :: !(V.Vector (Matrix R)), cbiases :: !(V.Vector R), cpadding :: !Int
}

instance Component CLayer where
    type Inp CLayer = V.Vector (Matrix R)
    type Out CLayer = V.Vector (Matrix R)
    -- trace is (input, convoluted output)
    newtype Trace CLayer = CTrace (Matrix R, Int, V.Vector (Matrix R))
    forwardT (CLayer{cfilters=fs, cbiases=bs, cpadding=p}) !inp =
        let !ov = V.zipWith feature fs bs
        in assert ok $ CTrace (inpsumup,V.length inp,ov)
      where
        !osize = let (w,_) = size (inp V.! 0)
                     (u,_) = size (fs  V.! 0)
                 in w+2*p-u+1
        !inpsumup = V.foldl1' add inp
        feature :: Matrix R -> R -> Matrix R
        feature f b = layerCorr2 p f inpsumup `add` konst b (osize,osize)
        ok = let
                 s0 = size (inp V.! 0) :: (Int,Int)
                 -- all channel are of the same size and are square
                 c1 = fst s0 == snd s0 && V.all ((==s0) . size) inp
                 t0 = size (fs  V.! 0) :: (Int,Int)
                 -- all filters are of the same size and are square
                 c2 = fst t0 == snd t0 && V.all ((==t0) . size) fs
             in c1 && c2
    output (CTrace (_,_,!a)) = a
    learn l (CTrace (!iv,!is,!av)) !odelta rate =
      let CLayer{cfilters=fs, cbiases=bs, cpadding=p} = l
          di :: Matrix R
          !di = V.foldl1' add $ V.zipWith (layerConv2 p) fs odelta
          idelta :: V.Vector (Matrix R)
          !idelta = V.replicate is di
          !dm = V.map (scale (negate rate) . layerCorr2 p iv) odelta
          !db = V.map ((* negate rate) . sumElements) odelta
      in (l{cfilters=V.zipWith add fs dm, cbiases=V.zipWith (+) bs db}, idelta)

data MaxPoolLayer = MaxPoolLayer Int

instance Component MaxPoolLayer where
  -- input are channels of matrix
  type Inp MaxPoolLayer = V.Vector (Matrix R)
  -- output are the same number of channels of pooled matrix
  type Out MaxPoolLayer = V.Vector (Matrix R)
  -- trace is (index of max in each pool, dimension of pools, pooled matrix)
  -- for each channel.
  newtype Trace MaxPoolLayer = PTrace (V.Vector ([[IndexOf Matrix]], (IndexOf Matrix), (Matrix R)))
  -- forward is to divide the input matrix in stride x stride sub matrices,
  -- and then find the max element in each sub matrices.
  forwardT (MaxPoolLayer stride) !inp = PTrace $ V.map pool inp
    where
      pool inp = let blks = toBlocksEvery stride stride inp :: [[Matrix R]]
                     !oi = map (map maxIndex) blks
                     !ov = fromLists $ zipWith (zipWith atIndex) blks oi
                 in (oi,size ov,ov)
  output (PTrace a) = V.map (\(_,_,o) ->o) a
  -- use the saved index-of-max in each pool to propagate the error.
  learn l@(MaxPoolLayer stride) (PTrace t) odelta _ = (l, V.zipWith gen t odelta)
    where
      gen (!iv,!si,_) od =
         let sub i v = assoc (stride, stride) 0 [(i,v)]
         in assert (si == size od) $ fromBlocks $ zipWith (zipWith sub) iv (toLists od)

-- Relu activator for single channel input
data ReluLayerS a = ReluLayerS
instance (Container c R, Multiplicable (c R)) => Component (ReluLayerS (c R)) where
    type Inp (ReluLayerS (c R)) = c R
    type Out (ReluLayerS (c R)) = c R
    newtype Trace (ReluLayerS (c R)) = ReluTrace (c R, c R)
    forwardT _ !inp = ReluTrace (inp, cmap relu inp)
    output (ReluTrace (_,!a)) = a
    learn a (ReluTrace (!iv,_)) !odelta _ = (a, odelta `hadamard` cmap relu' iv)

-- Relu activator for multiple channels input
data ReluLayerM a = ReluLayerM
instance (Container c R, Multiplicable (c R)) => Component (ReluLayerM (c R)) where
    type Inp (ReluLayerM (c R)) = V.Vector (c R)
    type Out (ReluLayerM (c R)) = V.Vector (c R)
    newtype Trace (ReluLayerM (c R)) = ReluTraceM (V.Vector (Trace (ReluLayerS (c R))))
    forwardT _ !inp = ReluTraceM $ V.map (forwardT ReluLayerS) inp
    output (ReluTraceM a) = V.map output a
    learn a (ReluTraceM ts) !odelta r = (a, V.zipWith (\t d -> snd $ learn ReluLayerS t d r) ts odelta)

-- Reshape from channels of matrix to a single vector
data ReshapeLayer = ReshapeLayer
instance Component ReshapeLayer where
  type Inp ReshapeLayer = V.Vector (Matrix R)
  type Out ReshapeLayer = Vector R
  newtype Trace ReshapeLayer = ReshapeTrace (Int, Int, Int, Vector R)
  forwardT _ !inp =
    let !b = V.length inp
        (!r,!c) = size (inp V.! 0)
        !o = V.foldr' (\x y -> flatten x SV.++ y) SV.empty inp
    in ReshapeTrace (b, r*c, c, o)
  output (ReshapeTrace (_,_,_,a)) = a
  learn a (ReshapeTrace (b,n,c,_)) !odelta _ =
    let !idelta = V.fromList $ map (reshape c) $ takesV (replicate b n) odelta
    in (a, idelta)

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

infixl 0 ++>
(++>) :: (Monad m, Component a, Component b, Component (a :+> b))
      => m a -> m b -> m (a :+> b)
(++>) = liftM2 (:+>)

newDLayer :: (Int, Int)         -- number of input channels, number of neurons (output channels)
          -> (R->R, R->R)       -- activate function and its derivative
          -> IO DLayer          -- new layer
newDLayer sz@(_,n) (af, af') =
    withSystemRandom . asGenIO $ \gen -> do
        w <- buildMatrix gen sz
        b <- return $ konst 1 n
        return $ DLayer w b af af'

newCLayer :: Int                -- number of filter
          -> Int                -- stride of each filter
          -> Int                -- number of padding
          -> IO CLayer
newCLayer nfilter sfilter npadding =
  withSystemRandom . asGenIO $ \gen -> do
      fs <- V.replicateM nfilter $ buildMatrix gen (sfilter, sfilter)
      bs <- return $ V.replicate nfilter 1
      return $ CLayer fs bs npadding

maxPool :: Int -> IO MaxPoolLayer
maxPool = return . MaxPoolLayer

reshape' :: IO ReshapeLayer
reshape' = return ReshapeLayer

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
layerCorr2 p k m = corr2d_s k padded
  where
    padded = fromBlocks [[z,0,0]
                        ,[0,m,0]
                        ,[0,0,z]]
    z = konst 0 (p,p)

layerConv2 :: Int -> Matrix R -> Matrix R -> Matrix R
layerConv2 p k m = conv2d_s k padded
  where
    padded = fromBlocks [[z,0,0]
                        ,[0,m,0]
                        ,[0,0,z]]
    z = konst 0 (p,p)

relu, relu' :: R-> R
relu = max 0
relu' x | x < 0     = 0
        | otherwise = 1

cost' a y | y == 1 && a >= y = 0
          | otherwise        = a - y

buildMatrix g (nr, nc) = do
  vals <- sequence (replicate (nr*nc) (double2Float <$> normal 0 0.01 g))
  return $ (nr >< nc) vals
