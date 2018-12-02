package com.huaze.shen.generator;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.Random;

/**
 * @author Huaze Shen
 * @date 2018-11-30
 */
public class DataGenerator {
    public void generate() {
        String trainFile = "src/main/resources/data/artificial/train-set.txt";
        int trainSize = 7000;
        generateData(trainFile, trainSize);

        String testFile = "src/main/resources/data/artificial/test-set.txt";
        int testSize = 3000;
        generateData(testFile, testSize);
    }

    private void generateData(String file, int size) {
        try {
            BufferedWriter bw = new BufferedWriter(new FileWriter(file));
            Random random = new Random();
            for (int i = 0; i < size / 2; i++) {
                String line = "";
                line += getRandomNumber(random);
                line += ",";
                line += getRandomNumber(random);
                line += ",";
                line += "1";
                line += "\n";
                bw.write(line);
            }
            for (int i = 0; i < size / 2; i++) {
                String line = "";
                line += -getRandomNumber(random);
                line += ",";
                line += -getRandomNumber(random);
                line += ",";
                line += "-1";
                line += "\n";
                bw.write(line);
            }
            bw.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private int getRandomNumber(Random random) {
        int randomNumber = random.nextInt(100);
        while (randomNumber == 0) {
            randomNumber = random.nextInt(100);
        }
        return randomNumber;
    }

    public static void main(String[] args) {
        DataGenerator dataGenerator = new DataGenerator();
        dataGenerator.generate();
    }
}
