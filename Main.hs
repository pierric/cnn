{-# LANGUAGE TypeFamilies, BangPatterns, TypeOperators, FlexibleContexts, FlexibleInstances #-}
module Main where

import CNN
import Parser1

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

main = mnistMain

mnistMain = debug3 -- makeMNIST >>= dotest

debug = do
    let n1 = CLayer (V.fromList $ [(3><3)[-0.2,-0.2,-0.2,0,0,0,0,0,0], (3><3)[0,0,0,0,0.2,0.2,0.2,0,0]])
                    (V.fromList $ [1,1])
                    1
    nx <- reluMulti
    n2 <- maxPool 2
    n3 <- reshape'
    -- n4 <- newDLayer (25, 4) (relu, relu') ++> newDLayer (4,  2) (relu, relu')
    n4 <- newDLayer (8,  2) (relu, relu')
    let nn = n1 :+> nx :+> n2 :+> n3 :+> n4
    let ds = V.singleton $ (4><4) [0,5,5,5
                                  ,5,5,5,0
                                  ,5,5,5,0
                                  ,5,5,5,0]
        ev = fromList [0.5,0.5] :: Vector CNN.R
    putStrLn $ replicate 40 '-'
    nn <- traceNN nn ds ev 1
    putStrLn $ replicate 40 '-'
    putStrLn "##forward"
    tracef nn ds
    putStrLn $ replicate 40 '-'
    nn <- traceNN nn ds ev 1
    putStrLn $ replicate 40 '-'
    putStrLn "##forward"
    tracef nn ds

debug2 = do
  let n1 = newCLayer 32 5 2 ++> reluMulti ++> maxPool 2
      n2 = newCLayer 64 5 2 ++> reluMulti ++> maxPool 2
      n3 = newDLayer (3136, 1024) (relu, relu') ++> newDLayer (1024,  10) (relu, relu')
      n4 = reshape' ++> newDLayer (196,30) (relu, relu') ++> newDLayer (30,10) (relu, relu')
  nn <- newCLayer 1 1 0 ++> reluMulti ++>maxPool 2 ++> n4
  putStrLn "Load training data."
  (ds,ev) <- head . uncurry zip <$> trainingData
  putStrLn "Load test data."
  nn <- foldM (\nn' _ -> traceNN nn' ds ev 0.001) nn [1..60]
  -- nn <- traceNN nn ds ev 0.001
  -- putStrLn $ replicate 40 '-'
  -- nn <- traceNN nn ds ev 0.001
  -- putStrLn $ replicate 40 '-'
  -- nn <- traceNN nn ds ev 0.001
  putStrLn $ "+" ++ prettyResult (forward nn ds)
  putStrLn $ "*" ++ prettyResult ev
  return ()

debug3 = do
  a0:a1:_ <- getArgs
  let cycle = read a0 :: Int
      rate  = read a1 :: Float
  let n1 = newCLayer 32 5 2 ++> reluMulti ++> maxPool 2
      n2 = newCLayer 64 5 2 ++> reluMulti ++> maxPool 2
      n3 = newDLayer (3136, 1024) (relu, relu') ++> newDLayer (1024,  10) (relu, relu')
      n4 = reshape' ++> newDLayer (392,30) (relu, relu') ++> newDLayer (30,10) (relu, relu')
  -- nn <- n1 ++> n2 ++> reshape' ++> n3
  nn <- newCLayer 32 5 2 ++> reluMulti ++>maxPool 2 ++>
        newCLayer 64 5 2 ++> reluMulti ++>maxPool 2 ++>
        reshape' ++> newDLayer (3136,1024) (relu, relu') ++> newDLayer (1024,10) (relu, relu')
  putStrLn "Load training data."
  dataset <- take 250 . uncurry zip <$> trainingData
  nn <- iterateM cycle (online rate dataset) nn
  flip mapM_ (take 12 dataset) $ \(ds,ev) -> do
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

makeMNIST = do
    let n1 = newCLayer 32 5 2 ++> reluMulti ++> maxPool 2
        n2 = newCLayer 64 5 2 ++> reluMulti ++> maxPool 2
        n3 = newDLayer (3136, 1024) (relu, relu') ++> newDLayer (1024,  10) (relu, relu')
    nn <- n1 ++> n2 ++> reshape' ++> n3
    putStrLn "Load training data."
    dataset <- take 500 . uncurry zip <$> trainingData
    putStrLn "Load test data."
    putStrLn "Learning."
    iterateM 2 (online dataset) nn

dotest :: (Component n, Inp n ~ Image, Out n ~ Label) => n -> IO ()
dotest !nn = do
    testset <- take 50 .uncurry zip <$> testData
    putStrLn "Start test"
    let result = map (postprocess . forward nn . fst) testset `using` parList rdeepseq
        expect = map (postprocess . snd) testset
        (co,wr)= partition (uncurry (==)) $ zip result expect
    putStrLn $ printf "correct: %d, wrong: %d" (length co) (length wr)

online = flip (foldl' $ learnStep (zipVectorWith cost') 0.0001)
iterateM :: Int -> (a -> a) -> a -> IO a
iterateM n f x = walk 0 x
  where
    walk !i !a | i == n    = return a
               | otherwise = do putStrLn ("Iteration " ++ show i)
                                walk (i+1) (f a)

postprocess :: Vector CNN.R -> Int
postprocess = fst . maximumBy cmp . zip [0..] . toList
  where cmp a b = compare (snd a) (snd b)