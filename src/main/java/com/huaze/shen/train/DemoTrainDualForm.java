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
    private final int epochs = 4;
    private final int TRAIN_SIZE = 7004;
    private final int FEATURE_DIMENSION = 2;
    private final double eta = 1.0;
    private double[] alphas;
    private List<Instance> trainSet;
    private String trainFile;
    private String modelFile;
    private double bias;

    public DemoTrainDualForm(String dataSetName) {
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
                alphas[instance.getId()] += eta;
                bias += eta * instance.getLabel();
            }
            if (getWrongPoints().isEmpty()) {
                break;
            }
        }

        double[] weights = new double[FEATURE_DIMENSION];
        for (int i = 0; i < FEATURE_DIMENSION; i++) {
            for (int j = 0; j < TRAIN_SIZE; j++) {
                weights[i] += alphas[j] * trainSet.get(j).getLabel() * trainSet.get(j).getFeature()[i];
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
        trainSet = new ArrayList<>();
        try {
            InputStream inputStream = DemoTrainDualForm.class.getClassLoader().getResourceAsStream(trainFile);
            BufferedReader br = new BufferedReader(new InputStreamReader(inputStream));
            String line;
            int i = 0;
            while ((line = br.readLine()) != null) {
                String[] lineSplit = line.split(",");
                double[] feature = new double[]{Double.valueOf(lineSplit[0]), Double.valueOf(lineSplit[1])};
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
        alphas = new double[TRAIN_SIZE];
        bias = 0;
    }

    private void writeModelFile() {
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
        String dataSetName = "artificial";
        DemoTrainDualForm demoTrainDualForm = new DemoTrainDualForm(dataSetName);
        demoTrainDualForm.train();
    }
}
