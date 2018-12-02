package com.huaze.shen.common;

import java.util.Arrays;

/**
 * @author Huaze Shen
 * @date 2018-11-30
 */
public class Instance {
    private int id;
    private double[] feature;
    private int label;

    public Instance(double[] feature, int label) {
        this.feature = feature;
        this.label = label;
    }

    public Instance(int id, double[] feature, int label) {
        this.id = id;
        this.feature = feature;
        this.label = label;
    }

    public double[] getFeature() {
        return feature;
    }

    public void setFeature(double[] feature) {
        this.feature = feature;
    }

    public int getLabel() {
        return label;
    }

    public void setLabel(int label) {
        this.label = label;
    }

    public int getId() {
        return id;
    }

    public void setId(int id) {
        this.id = id;
    }

    @Override
    public String toString() {
        return "Instance{" +
                "id=" + id +
                ", feature=" + Arrays.toString(feature) +
                ", label=" + label +
                '}';
    }
}
