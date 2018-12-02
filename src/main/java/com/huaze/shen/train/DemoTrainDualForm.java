package com.huaze.shen.train;

import com.huaze.shen.common.Instance;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.List;

/**
 * 感知机对偶形式
 *
 * @author Huaze Shen
 * @date 2018-11-30
 */
public class DemoTrainDualForm {
    private final int TRAIN_SIZE = 7000;
    private final int FEATURE_DIMENSION = 2;
    private final double eta = 1.0;
    private double[] alphas;
    private List<Instance> trainSet;
    private double bias;

    public DemoTrainDualForm() {
        init();
    }

    public void train() {
        while (!getWrongPoints().isEmpty()) {
            List<Instance> wrongPoints = getWrongPoints();
            Instance selectedWrongPoint = wrongPoints.get(0);
            alphas[selectedWrongPoint.getId()] += eta;
            bias += eta * selectedWrongPoint.getLabel();
        }
        String modelFile = "src/main/resources/alpha.txt";
        writeModelFile(modelFile);
    }

    private List<Instance> getWrongPoints() {
        List<Instance> instances = new ArrayList<>();
        for (Instance instance : trainSet) {
            if (isWrong(instance)) {
                instances.add(instance);
            }
        }
        return instances;
    }

    private boolean isWrong(Instance instance) {
        int label = instance.getLabel();
        double predict = 0;
        for (int i = 0; i < TRAIN_SIZE; i++) {
            Instance eachInstance = trainSet.get(i);
            predict += alphas[i] * eachInstance.getLabel() * calculateInnerProduct(instance, eachInstance);
        }
        predict += bias;
        return label * predict <= 0;
    }

    private double calculateInnerProduct(Instance currentInstance, Instance eachInstance) {
        double predict = 0;
        for (int i = 0; i < FEATURE_DIMENSION; i++) {
            predict += eachInstance.getFeature()[i] * currentInstance.getFeature()[i];
        }
        return predict;
    }

    private void init() {
        loadDataSet();
        initModel();
    }

    private void loadDataSet() {
        String trainFile = "src/main/resources/data/train-set.txt";
        trainSet = new ArrayList<>();
        try {
            BufferedReader br = new BufferedReader(new FileReader(trainFile));
            String line;
            int i = 0;
            while ((line = br.readLine()) != null) {
                String[] lineSplit = line.split(",");
                int[] feature = new int[]{Integer.valueOf(lineSplit[0]), Integer.valueOf(lineSplit[1])};
                int label = Integer.valueOf(lineSplit[2]);
                trainSet.add(new Instance(i, feature, label));
                i += 1;
            }
            br.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void initModel() {
        alphas = new double[]{-1, 1};
        bias = 0;
    }

    private void writeModelFile(String modelFile) {
        try {
            BufferedWriter bw = new BufferedWriter(new FileWriter(modelFile));
            StringBuilder alphaLine = new StringBuilder();
            for (int i = 0; i < TRAIN_SIZE; i++) {
                alphaLine.append(alphas[i]);
                alphaLine.append(",");
            }
            bw.write(alphaLine.substring(0, alphaLine.length() - 1));
            bw.write("\n");
            bw.write(String.valueOf(bias));
            bw.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        DemoTrainDualForm demoTrainDualForm = new DemoTrainDualForm();
        demoTrainDualForm.train();
    }
}
