package com.huaze.shen.predict;

import com.huaze.shen.common.Instance;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * 在测试集上预测（感知机原始形式）
 *
 * @author Huaze Shen
 * @date 2018-11-30
 */
public class DemoPredict {
    private final int featureDimension;
    private List<Instance> testSet;
    private double[] weights;
    private double bias;
    private String testFile;
    private String modelFile;

    public DemoPredict(String dataSetName, int featureDimension) {
        this.featureDimension = featureDimension;
        testFile = "data/" + dataSetName + "/test-set.txt";
        //testFile = "data/" + dataSetName + "/train-set.txt";
        modelFile = "src/main/resources/model/"+ dataSetName + "/model.txt";
        init();
    }

    public void predict() {
        int wrongCount = 0;
        for (Instance instance : testSet) {
            int goldLabel = instance.getLabel();
            int predictLabel = predictEachInstance(instance);
            if (goldLabel != predictLabel) {
                wrongCount += 1;
            }
            System.out.println("feature: " + Arrays.toString(instance.getFeature()));
            System.out.println("label: " + goldLabel);
            System.out.println("predict: " + predictLabel);
            System.out.println();
        }
        System.out.println("wrong count: " + wrongCount);
    }

    private int predictEachInstance(Instance instance) {
        double y = 0;
        for (int i = 0; i < featureDimension; i++) {
            y += instance.getFeature()[i] * weights[i];
        }
        y += bias;
        if (y <= 0) {
            return -1;
        }
        return 1;
    }

    private void init() {
        loadValidSet();
        loadModel();
    }

    private void loadValidSet() {
        testSet = new ArrayList<>();
        try {
            InputStream inputStream = DemoPredict.class.getClassLoader().getResourceAsStream(testFile);
            BufferedReader br = new BufferedReader(new InputStreamReader(inputStream));
            String line;
            while ((line = br.readLine()) != null) {
                String[] lineSplit = line.split(",");
                double[] feature = new double[featureDimension];
                for (int i = 0; i < featureDimension; i++) {
                    feature[i] = Double.valueOf(lineSplit[i]);
                }
                int label = Integer.valueOf(lineSplit[featureDimension]);
                testSet.add(new Instance(feature, label));
            }
            br.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void loadModel() {
        try {
            BufferedReader br = new BufferedReader(new FileReader(modelFile));
            String weightLine = br.readLine().trim();
            String[] weightLineSplit = weightLine.split(",");
            weights = new double[featureDimension];
            for (int i = 0; i < featureDimension; i++) {
                weights[i] = Double.valueOf(weightLineSplit[i]);
            }
            String biasLine = br.readLine().trim();
            bias = Double.valueOf(biasLine);
            br.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        //String dataSetName = "artificial";
        String dataSetName = "sonar";
        //int featureSize = 2;
        int featureSize = 60;
        DemoPredict demoPredict = new DemoPredict(dataSetName, featureSize);
        demoPredict.predict();
    }
}
