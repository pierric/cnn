{-# LANGUAGE TypeFamilies, BangPatterns #-}
module Main where

import CNN
import Parser1

import Numeric.LinearAlgebra
import Numeric.LinearAlgebra.Data
import Numeric.LinearAlgebra.Devel
import Data.List(foldl',partition,maximumBy)
import Text.Printf (printf)

main = mnistMain

mnistMain = makeMNIST >>= dotest

makeMNIST = do
    let n1 = newCLayer 32 5 2 ++> maxPool 2
        n2 = newCLayer 64 5 2 ++> maxPool 2
        n3 = newDLayer (6272, 30) (relu, relu') ++> newDLayer (30,  10) (relu, relu')
        n4 = newDLayer (3136, 1024) (relu, relu') ++> newDLayer (1024,  10) (relu, relu')
        n5 = newDLayer (784, 30) (relu, relu') ++> newDLayer (30,  10) (relu, relu')
    --nn <- n1 ++> reshape' ++> n3
    --nn <- reshape' ++> n5
    nn <- n1 ++> n2 ++> reshape' ++> n4

    putStrLn "Load training data."
    dataset <- take 1000 . uncurry zip <$> trainingData
    putStrLn "Load test data."
    putStrLn "Learning."
    iterateM 2 (online dataset) nn

online = flip (foldl' $ learnStep (zipVectorWith cost') 0.002)
iterateM :: Int -> (a -> a) -> a -> IO a
iterateM n f x = walk 0 x
  where
    walk !i !a | i == n    = return a
               | otherwise = do putStrLn ("Iteration " ++ show i)
                                walk (i+1) (f a)

postprocess :: Vector CNN.R -> Int
postprocess = fst . maximumBy cmp . zip [0..] . toList
  where cmp a b = compare (snd a) (snd b)

dotest :: (Component n, Inp n ~ Image, Out n ~ Label) => n -> IO ()
dotest !nn = do
    testset <- uncurry zip <$> testData
    putStrLn "Start test"
    let result = map (postprocess . forward nn . fst) testset
        expect = map (postprocess . snd) testset
        (co,wr)= partition (uncurry (==)) $ zip result expect
    putStrLn $ printf "correct: %d, wrong: %d" (length co) (length wr)
