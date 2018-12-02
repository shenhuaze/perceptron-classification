package com.huaze.shen.train;

import com.huaze.shen.common.Instance;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * 感知机学习算法的对偶形式
 *
 * @author Huaze Shen
 * @date 2018-11-30
 */
public class DemoTrainDualForm {
    private final int epochs = 50000;
    private int trainSize;
    private int featureDimension;
    private final double eta = 1.0;
    private double[] alphas;
    private double[] weights;
    private List<Instance> trainSet;
    private String trainFile;
    private String alphaFile;
    private String modelFile;
    private double bias;

    public DemoTrainDualForm(String dataSetName, int featureDimension, int trainSize) {
        this.featureDimension = featureDimension;
        this.trainSize = trainSize;
        trainFile = "data/" + dataSetName + "/train-set.txt";
        alphaFile = "src/main/resources/model/" + dataSetName + "/alpha.txt";
        modelFile = "src/main/resources/model/" + dataSetName + "/model.txt";
        init();
    }

    public void train() {
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (Instance instance : trainSet) {
                if (!isWrong(instance)) {
                    continue;
                }
                alphas[instance.getId()] += eta;
                bias += eta * instance.getLabel();
            }
            List<Instance> wrongPoints = getWrongPoints();
            if (epoch % 100 == 0) {
                System.out.println("epoch: " + epoch);
                System.out.println("wrong count: " + wrongPoints.size());
            }
            if (wrongPoints.isEmpty()) {
                break;
            }
        }

        weights = new double[featureDimension];
        for (int i = 0; i < featureDimension; i++) {
            for (int j = 0; j < trainSize; j++) {
                weights[i] += alphas[j] * trainSet.get(j).getLabel() * trainSet.get(j).getFeature()[i];
            }
        }
        System.out.println("weights: " + Arrays.toString(weights));
        System.out.println("bias: " + bias);

        writeAlphaFile();
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
        for (int i = 0; i < trainSize; i++) {
            Instance eachInstance = trainSet.get(i);
            predict += alphas[i] * eachInstance.getLabel() * calculateInnerProduct(instance, eachInstance);
        }
        predict += bias;
        return label * predict <= 0;
    }

    private double calculateInnerProduct(Instance currentInstance, Instance eachInstance) {
        double predict = 0;
        for (int i = 0; i < featureDimension; i++) {
            predict += eachInstance.getFeature()[i] * currentInstance.getFeature()[i];
        }
        return predict;
    }

    private void init() {
        loadDataSet();
        initModel();
    }

    private void loadDataSet() {
        trainSet = new ArrayList<>();
        try {
            InputStream inputStream = DemoTrainDualForm.class.getClassLoader().getResourceAsStream(trainFile);
            BufferedReader br = new BufferedReader(new InputStreamReader(inputStream));
            String line;
            int i = 0;
            while ((line = br.readLine()) != null) {
                String[] lineSplit = line.split(",");
                double[] feature = new double[featureDimension];
                for (int j = 0; j < featureDimension; j++) {
                    feature[j] = Double.valueOf(lineSplit[j]);
                }
                int label = Integer.valueOf(lineSplit[featureDimension]);
                trainSet.add(new Instance(i, feature, label));
                i += 1;
            }
            br.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void initModel() {
        alphas = new double[trainSize];
        bias = 0;
    }

    private void writeAlphaFile() {
        try {
            BufferedWriter bw = new BufferedWriter(new FileWriter(alphaFile));
            for (int i = 0; i < trainSize; i++) {
                if (alphas[i] != 0) {
                    StringBuilder alphaLine = new StringBuilder();
                    alphaLine.append(alphas[i]);
                    alphaLine.append("\t");
                    alphaLine.append(trainSet.get(i).getLabel());
                    alphaLine.append("\t");
                    double[] feature = trainSet.get(i).getFeature();
                    for (double featureEachDimension : feature) {
                        alphaLine.append(featureEachDimension);
                        alphaLine.append(",");
                    }
                    bw.write(alphaLine.substring(0, alphaLine.length() - 1) + "\n");
                }
            }
            bw.write(String.valueOf(bias));
            bw.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void writeModelFile() {
        try {
            BufferedWriter bw = new BufferedWriter(new FileWriter(modelFile));
            StringBuilder weightsLine = new StringBuilder();
            for (double weight : weights) {
                weightsLine.append(weight);
                weightsLine.append(",");
            }
            bw.write(weightsLine.substring(0, weightsLine.length() - 1));
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
        //int trainSize = 7004;
        int trainSize = 146;
        DemoTrainDualForm demoTrainDualForm = new DemoTrainDualForm(dataSetName, featureDimension, trainSize);
        demoTrainDualForm.train();
    }
}
