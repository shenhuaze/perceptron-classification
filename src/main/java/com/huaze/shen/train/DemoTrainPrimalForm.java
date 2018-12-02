package com.huaze.shen.train;

import com.huaze.shen.common.Instance;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * 感知机学习算法的原始形式
 *
 * @author Huaze Shen
 * @date 2018-11-30
 */
public class DemoTrainPrimalForm {
    private final int epochs = 5;
    private final double eta = 1.0;
    private int featureDimension;
    private String trainFile;
    private List<Instance> trainSet;
    private String modelFile;
    private double[] weights;
    private double bias;

    public DemoTrainPrimalForm(String dataSetName, int featureDimension) {
        this.featureDimension = featureDimension;
        trainFile = "data/" + dataSetName + "/train-set.txt";
        modelFile = "src/main/resources/model/"+ dataSetName + "/model.txt";
        init();
    }

    public void train() {
        for (int epoch = 0; epoch < epochs; epoch++) {
            System.out.println("epoch: " + epoch);
            for (Instance instance : trainSet) {
                if (!isWrong(instance)) {
                    continue;
                }
                for (int i = 0; i < featureDimension; i++) {
                    weights[i] += eta * instance.getLabel() * instance.getFeature()[i];
                }
                bias += eta * instance.getLabel();
            }
            if (getWrongPoints().isEmpty()) {
                break;
            }
        }
        System.out.println("weights: " + Arrays.toString(weights));
        System.out.println("bias: " + bias);
        writeModelFile();
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
        for (int i = 0; i < featureDimension; i++) {
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
        trainSet = new ArrayList<>();
        try {
            InputStream inputStream = DemoTrainPrimalForm.class.getClassLoader().getResourceAsStream(trainFile);
            BufferedReader br = new BufferedReader(new InputStreamReader(inputStream));
            String line;
            while ((line = br.readLine()) != null) {
                String[] lineSplit = line.split(",");
                double[] feature = new double[featureDimension];
                for (int i = 0; i < featureDimension; i++) {
                    feature[i] = Double.valueOf(lineSplit[i]);
                }
                int label = Integer.valueOf(lineSplit[featureDimension]);
                trainSet.add(new Instance(feature, label));
            }
            br.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void initWeights() {
        weights = new double[featureDimension];
        bias = 0;
    }

    private void writeModelFile() {
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
        //String dataSetName = "artificial";
        String dataSetName = "sonar";
        //int featureDimension = 2;
        int featureDimension = 60;
        DemoTrainPrimalForm demoTrainPrimalForm = new DemoTrainPrimalForm(dataSetName, 60);
        demoTrainPrimalForm.train();
    }
}
