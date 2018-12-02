package com.huaze.shen.train;

import com.huaze.shen.common.Instance;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.List;

/**
 * 感知机原始形式
 *
 * @author Huaze Shen
 * @date 2018-11-30
 */
public class DemoTrainPrimalForm {
    private final int FEATURE_DIMENSION = 2;
    private final double eta = 1.0;
    private List<Instance> trainSet;
    private double[] weights;
    private double bias;

    public DemoTrainPrimalForm() {
        init();
    }

    public void train() {
        while (!getWrongPoints().isEmpty()) {
            List<Instance> wrongPoints = getWrongPoints();
            Instance selectedWrongPoint = wrongPoints.get(0);
            for (int i = 0; i < FEATURE_DIMENSION; i++) {
                weights[i] += eta * selectedWrongPoint.getLabel() * selectedWrongPoint.getFeature()[i];
            }
            bias += eta * selectedWrongPoint.getLabel();
        }
        String modelFile = "src/main/resources/model.txt";
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
        for (int i = 0; i < FEATURE_DIMENSION; i++) {
            predict += instance.getFeature()[i] * weights[i];
        }
        predict += bias;
        return label * predict <= 0;
    }

    private void init() {
        loadDataSet();
        initWeights();
    }

    private void loadDataSet() {
        String trainFile = "src/main/resources/data/train-set.txt";
        trainSet = new ArrayList<>();
        try {
            BufferedReader br = new BufferedReader(new FileReader(trainFile));
            String line;
            while ((line = br.readLine()) != null) {
                String[] lineSplit = line.split(",");
                int[] feature = new int[]{Integer.valueOf(lineSplit[0]), Integer.valueOf(lineSplit[1])};
                int label = Integer.valueOf(lineSplit[2]);
                trainSet.add(new Instance(feature, label));
            }
            br.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void initWeights() {
        weights = new double[]{0, 0};
        bias = 0;
    }

    private void writeModelFile(String modelFile) {
        try {
            BufferedWriter bw = new BufferedWriter(new FileWriter(modelFile));
            bw.write(weights[0] + "," + weights[1]);
            bw.write("\n");
            bw.write(String.valueOf(bias));
            bw.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        DemoTrainPrimalForm demoTrainPrimalForm = new DemoTrainPrimalForm();
        demoTrainPrimalForm.train();
    }
}
