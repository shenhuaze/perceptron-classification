package com.huaze.shen.predict;

import com.huaze.shen.common.Instance;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * @author Huaze Shen
 * @date 2018-11-30
 */
public class DemoPredictPrimalForm {
    private final int FEATURE_DIMENSION = 2;
    private List<Instance> validSet;
    private double[] weights;
    private double bias;

    public DemoPredictPrimalForm() {
        init();
    }

    public void predict() {
        int wrongCount = 0;
        for (Instance instance : validSet) {
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
        System.out.println(wrongCount);
    }

    private int predictEachInstance(Instance instance) {
        double y = 0;
        for (int i = 0; i < FEATURE_DIMENSION; i++) {
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
        String trainFile = "src/main/resources/data/test-set.txt";
        validSet = new ArrayList<>();
        try {
            BufferedReader br = new BufferedReader(new FileReader(trainFile));
            String line;
            while ((line = br.readLine()) != null) {
                String[] lineSplit = line.split(",");
                double[] feature = new double[]{Double.valueOf(lineSplit[0]), Double.valueOf(lineSplit[1])};
                int label = Integer.valueOf(lineSplit[2]);
                validSet.add(new Instance(feature, label));
            }
            br.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void loadModel() {
        String modelFile = "src/main/resources/model.txt";
        try {
            BufferedReader br = new BufferedReader(new FileReader(modelFile));
            String weightLine = br.readLine().trim();
            weights = new double[2];
            weights[0] = Double.valueOf(weightLine.split(",")[0]);
            weights[1] = Double.valueOf(weightLine.split(",")[1]);
            String biasLine = br.readLine().trim();
            bias = Double.valueOf(biasLine);
            br.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        DemoPredictPrimalForm demoPredictPrimalForm = new DemoPredictPrimalForm();
        demoPredictPrimalForm.predict();
    }
}
