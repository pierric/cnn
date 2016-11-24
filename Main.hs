{-# LANGUAGE TypeFamilies, BangPatterns, TypeOperators, FlexibleContexts, FlexibleInstances #-}
module Main where

import CNN
import Parser

import Numeric.LinearAlgebra
import Numeric.LinearAlgebra.Data
import Numeric.LinearAlgebra.Devel
import qualified Data.Vector as V
import Data.List(foldl',partition,maximumBy)
import Text.Printf (printf)
import Control.Parallel.Strategies
import System.Environment
import Text.PrettyPrint.Free hiding (flatten)
import Control.Monad

main = makeMNIST >>= dotest

makeMNIST = do
    nn <- newCLayer 1 4 7 3 ++> reluMulti ++> maxPool 2 ++>
          reshapeL ++>
          newDLayer (784,256) (relu, relu') ++>
          newDLayer (256,64)  (relu, relu') ++>
          newDLayer (64,10)   (relu, relu')
    putStrLn "Load training data."
    dataset <- uncurry zip <$> trainingData
    putStrLn "Load test data."
    putStrLn "Learning."
    iterateM 30 (online dataset) nn

dotest :: (Component n, Inp n ~ Image, Out n ~ Label) => n -> IO ()
dotest !nn = do
    testset <- uncurry zip <$> testData
    putStrLn "Start test"
    let result = map (postprocess . forward nn . fst) testset `using` parList rdeepseq
        expect = map (postprocess . snd) testset
        (co,wr)= partition (uncurry (==)) $ zip result expect
    putStrLn $ printf "correct: %d, wrong: %d" (length co) (length wr)

online ds !nn = walk ds nn
  where
    walk []     !nn = nn
    walk (d:ds) !nn = let !nn' = learnStep (zipVectorWith cost') 0.0010 nn d
                      in walk ds nn'

iterateM :: Int -> (a -> a) -> a -> IO a
iterateM n f x = walk 0 x
  where
    walk !i !a | i == n    = return a
               | otherwise = do -- when (i `mod` 10 == 0) $ putStrLn ("Iteration " ++ show i)
                                putStrLn ("Iteration " ++ show i)
                                walk (i+1) $! f a

postprocess :: Vector CNN.R -> Int
postprocess = fst . maximumBy cmp . zip [0..] . toList
  where cmp a b = compare (snd a) (snd b)

--
-- for debugging purpose
--
debug1 = do
    let n1 = CLayer (V.singleton $ V.fromList $ [(3><3)[-0.2,-0.2,-0.2,0,0,0,0,0,0], (3><3)[0,0,0,0,0.2,0.2,0.2,0,0]])
                    (V.fromList $ [0.1,0.1])
                    1
    nx <- reluMulti
    n2 <- maxPool 2
    n3 <- reshapeL
    -- n4 <- newDLayer (25, 4) (relu, relu') ++> newDLayer (4,  2) (relu, relu')
    n4 <- newDLayer (8,  2) (relu, relu')
    let nn = n1 :+> nx :+> n2 :+> n3 :+> n4
    let ds = V.singleton $ (4><4) [0,5,5,5
                                  ,5,5,5,0
                                  ,5,5,5,0
                                  ,5,5,5,0]
        ev = fromList [0.2,0.8] :: Vector CNN.R
    let rate = 0.02
    putStrLn $ replicate 40 '-'
    nn <- traceNN nn ds ev rate
    putStrLn $ replicate 40 '-'
    nn <- traceNN nn ds ev rate
    putStrLn $ replicate 40 '-'
    nn <- traceNN nn ds ev rate
    putStrLn $ replicate 40 '-'
    nn <- traceNN nn ds ev rate
    putStrLn $ replicate 40 '-'
    nn <- traceNN nn ds ev rate
    putStrLn $ replicate 40 '-'
    nn <- traceNN nn ds ev rate
    putStrLn $ replicate 40 '-'

debug2 = do
  nn <- newCLayer 1 2 3 1 ++> reluMulti ++> maxPool 2 ++>
        newCLayer 2 4 3 1 ++> reluMulti ++> maxPool 2 ++>
        reshapeL ++> newDLayer (196,30) (relu, relu') ++> newDLayer (30,10) (relu, relu')
  putStrLn "Load training data."
  dataset <- uncurry zip <$> trainingData
  let ts = take 120 dataset
  putStrLn "Load test data."
  nn <- foldM (\x _ -> do
          foldM (\x (i,(ds,ev)) -> do
            traceERR x ds ev 0.002 ("D" ++ show i)) x (zip [1..] ts)) nn [1..400]
  flip mapM_ (take 10 dataset) $ \(ds,ev) -> do
    putStrLn $ "+" ++ prettyResult (forward nn ds)
    putStrLn $ "*" ++ prettyResult ev

debug3 = do
  a0:a1:_ <- getArgs
  let cycle = read a0 :: Int
      rate  = read a1 :: Float
  -- nn <- newCLayer 1  16 5 2 ++> reluMulti ++> maxPool 2 ++>
  --       newCLayer 16 32 5 2 ++> reluMulti ++> maxPool 2 ++>
  --       reshapeL ++> newDLayer (1568,1024) (relu, relu') ++> newDLayer (1024,10) (relu, relu')
  nn <- newCLayer 1 4 7 3 ++> reluMulti ++> maxPool 2 ++>
        -- newCLayer 4 8 7 3 ++> reluMulti ++> maxPool 2 ++>
        reshapeL ++> newDLayer (784,256) (relu, relu') ++> newDLayer (256,10) (relu, relu')
  putStrLn "Load training data."
  dataset <- uncurry zip <$> trainingData
  nn <- iterateM cycle (online rate dataset) nn
  testset <- uncurry zip <$> testData
  flip mapM_ (take 10 testset) $ \(ds,ev) -> do
    putStrLn $ "+" ++ prettyResult (forward nn ds)
    putStrLn $ "*" ++ prettyResult ev
  where
    online :: (Component a, Inp a ~ Image, Out a ~ Label) => CNN.R -> [(Image, Label)] -> a -> a
    online rate = flip (foldl' $ learnStep (zipVectorWith cost') rate)
showPretty x = displayS (renderPretty 0.4 500 x) ""
prettyResult a =
  showPretty $ text (printf "%02d:" (postprocess a)) <+> pretty a
class (Component a, Show (Inp a), Pretty (Out a)) => Debug a where
  tracef :: a -> Inp a -> IO (Trace a)
  tracef a iv = do
    let ao = forwardT a iv
    putStrLn $ showPretty $ pretty $ output $ ao
    return $ ao
  traceb :: a -> Trace a -> Out a -> CNN.R -> IO a
  traceb a t d r = do
    let (a',t') = learn a t d r
    putStrLn $ show $ t'
    return a'
instance (Debug a, Debug b, Out a ~ Inp b) => Debug (a :+> b) where
  tracef (a :+> b) iv = do
    let ao = forwardT a iv
    putStrLn $ showPretty $ pretty $ output $ ao
    bo <- tracef b $ output $ ao
    return $ TTrace (bo, ao)
  traceb (a :+> b) (TTrace (bo, ao)) bd rate = do
    let (b',ad) = learn b bo bd rate
    putStrLn $ show $ ad
    a' <- traceb a ao ad rate
    return $ a' :+> b'
instance Debug (ReluLayerM (Matrix CNN.R))
instance Debug ReshapeLayer
instance Debug MaxPoolLayer
instance Debug DLayer
instance Debug CLayer

traceNN nn iv ev rt = do
  putStrLn $ show nn
  putStrLn "##forward"
  ot  <- tracef nn iv
  putStrLn "##backward"
  let err = cost' (output ot) ev
  putStrLn $ "##Err:" ++ show err
  nn' <- traceb nn ot err rt
  putStrLn $ "##updated network"
  putStrLn $ show nn'
  return nn'

traceERR nn iv ev rt lbl = do
  putStrLn $ lbl ++ "##forward"
  let ot = forwardT nn iv
  putStrLn $ lbl ++ "##backward"
  let err = cost' (output ot) ev
  putStrLn $ lbl ++ "##Err:" ++ show err
  let nn' = fst $ learn nn ot err rt
  return nn'
