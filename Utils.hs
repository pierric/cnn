{-# LANGUAGE BangPatterns, FlexibleInstances, FlexibleContexts #-}
module Utils where
import Numeric.LinearAlgebra
import Numeric.LinearAlgebra.Devel
import Numeric.GSL.Fourier
import Data.Complex
import Control.Exception
import Control.DeepSeq
import Control.Parallel
import Control.Parallel.Strategies
import Control.Monad
import qualified Data.Vector as VecB
import qualified Data.Vector.Generic as VecGeneric
import qualified Data.Vector.Fusion.Bundle as VecFusion
import qualified Data.Vector.Fusion.Bundle.Monadic as VecFusionM
import qualified Data.Vector.Storable as VecS
import qualified Data.Vector.Storable.Mutable as VecM

fft2d :: Matrix (Complex Double) -> Matrix (Complex Double)
fft2d m = let !x = fromRows $ map fft $ unsafeToRows m
              !y = fromColumns $ map fft $ toColumns x
          in y
ifft2d :: Matrix (Complex Double) -> Matrix (Complex Double)
ifft2d m = let !x = fromRows $ map ifft $ unsafeToRows m
               !y = fromColumns $ map ifft $ toColumns x
           in y

-- fft2d :: Matrix (Complex Double) -> Matrix (Complex Double)
-- fft2d m = let rh:rr = map fft $ force $ toRows m
--               x = fromRows $ withStrategy (parList rdeepseq) rr `pseq` (rh : rr)
--               sh:ss = map fft $ force $ toColumns x
--               y = fromColumns $ withStrategy (parList rdeepseq) ss `pseq` (sh : ss)
--           in y
--
-- ifft2d :: Matrix (Complex Double) -> Matrix (Complex Double)
-- ifft2d m = let rh:rr = map fft $ force $ toRows m
--                x = fromRows $ withStrategy (parList rdeepseq) rr `pseq` (rh : rr)
--                sh:ss = map fft $ force $ toColumns x
--                y = fromColumns $ withStrategy (parList rdeepseq) ss `pseq` (sh : ss)
--            in y

conv2d_b :: (Numeric t, ConvFD t, Container Vector t, Container Matrix t)
         => Matrix t -> Matrix t -> Matrix t
conv2d_b !k !m | w1 > w2 && h1 < h2 = error "convolution cannot be performed"
               | w1 < w2 && h1 > h2 = error "convolution cannot be performed"
               | w1 > w2   = conv2d_b m k
               | otherwise =
    -- convolution via FFT is actually cyclic conv. so we need to 0-pad the
    -- matrix being convoluted by half the size of the kernel matrix.
    -- the kernel matrix is also 0-padded to be the equal size.
    -- finall, extra
    let !hw = w1 `div` 2
        !hh = h1 `div` 2
        !z1 = konst 0 (w2-w1+hw,h2-h1+hh)
        !z2 = konst 0 (hw, hh)
        m1' = fft2d $ fromBlocks [[m1,0],[0,z1]]
        m2' = fft2d $ fromBlocks [[m2,0],[0,z2]]
        mr  = ifft2d $ (force m1' `par` (force m2' `pseq` hadamard m1' m2'))
        ms  = subMatrix (w1-1,h1-1) (w2-w1+1,h2-h1+1) $ fst . fromComplex $ mr
    in fromDouble $ ms
  where
     !m1 = complex $ toDouble k
     !m2 = complex $ toDouble m
     (w1,h1) = size m1
     (w2,h2) = size m2

corr2d_b k m | w > s && h < t = error "convolution cannot be performed"
             | w < s && h > t = error "convolution cannot be performed"
             | w > s     = conv2d_b (fliprl $ flipud m) k
             | otherwise = conv2d_b (fliprl $ flipud k) m
  where
    (w,h)   = size k
    (s,t)   = size m

corr2d_s :: (Numeric t, Container Vector t, Container Matrix t)
         => Matrix t -> Matrix t -> Matrix t
corr2d_s k m | w > s && h < t = error "correlation cannot be performed"
             | w < s && h > t = error "correlation cannot be performed"
             | w > s = corr2d_s m k
             | otherwise = final
  where
    (w,h) = size k
    (s,t) = size m
    (u,v) = (s-w+1, t-h+1)
    subs  = map (\s->subMatrix s (w,h) m) $ [(x,y) | x<-[0..u-1], y<-[0..v-1]]
    -- use unsafe* methods to create the intermediate matrix fast.
    t_rows = u*v
    t_cols = w*h
    !transformed = matrixFromVector RowMajor t_rows t_cols $ VecS.create $ do
        mat <- VecM.new (t_rows*t_cols)
        forM_ (zip [0..] subs) $ \(ri,rm) -> do
            let bs = t_cols*ri
            forM_ (zip [0..] $ unsafeToRows rm) $ \(ci, rv) -> do
                let tv = VecM.unsafeSlice (bs+h*ci) h mat
                {-# SCC "corr-data-copy" #-} VecS.unsafeCopy tv rv
        return mat
    !weights = flatten k
    !final   = reshape v $ transformed #> weights

conv2d_s k m | w > s && h < t = error "convolution cannot be performed"
             | w < s && h > t = error "convolution cannot be performed"
             | w > s     = corr2d_s (fliprl $ flipud m) k
             | otherwise = corr2d_s (fliprl $ flipud k) m
  where
    (w,h)   = size k
    (s,t)   = size m

class ConvFD t where
  fromDouble :: Matrix Double -> Matrix t
  toDouble   :: Matrix t -> Matrix Double
instance ConvFD Double where
  fromDouble = id
  toDouble = id
instance ConvFD Float where
  fromDouble = single
  toDouble = double

class Multiplicable c where
    hadamard :: c -> c -> c
instance (Num a, Element a) => Multiplicable (Vector a)
  where
    hadamard a b = zipVectorWith (*) a b
instance (Num a, Element a) => Multiplicable (Matrix a)
  where
    hadamard a b = reshape (cols a) (flatten a `hadamard` flatten b)

-- parallel :: NFData a => VecB.Vector a -> VecB.Vector a
-- parallel vec =
--     VecB.fromList (parlist $ VecB.toList vec)
--   where
--     parlist []     = []
--     parlist [a]    = [a]
--     parlist (a:as) = (as `using` parList rdeepseq) `pseq` a:as

parallel :: NFData a => VecB.Vector a -> VecB.Vector a
parallel vec = (VecB.tail vec `using` parvec) `pseq` vec
  where
    parvec = VecB.mapM (rparWith rdeepseq)
