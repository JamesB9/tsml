package ml_6002b_coursework;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.NumericToBinary;
import weka.filters.unsupervised.instance.RemoveWithValues;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.Enumeration;

/**
 * Interface for alternative attribute split measures for Part 2.2 of the coursework
 */
public abstract class AttributeSplitMeasure {

    public abstract double computeAttributeQuality(Instances data, Attribute att) throws Exception;

    /**
     * Splits a dataset according to the values of a nominal attribute.
     *
     * @param data the data which is to be split
     * @param att the attribute to be used for splitting
     * @return the sets of instances produced by the split
     */
    public Instances[] splitData(Instances data, Attribute att) {
        Instances[] splitData = new Instances[att.numValues()];
        for (int i = 0; i < att.numValues(); i++) {
            splitData[i] = new Instances(data, data.numInstances());
        }

        for (Instance inst: data) {
            splitData[(int) inst.value(att)].add(inst);
        }

        for (Instances split : splitData) {
            split.compactify();
        }

        return splitData;
    }


    public Instances[] splitDataOnNumeric(Instances data, Attribute att, double splitValue) throws Exception {
        /*
        Instances[] splitData = new Instances[2];
        for (int i = 0; i < att.numValues(); i++) {
            splitData[i] = new Instances(data, data.numInstances());
        }

        for (Instance instance : data) {
            if(instance.value(att) < splitValue) splitData[0].add(instance);
            else splitData[1].add(instance);
        }

        splitData[0].compactify();
        splitData[1].compactify();

        return splitData;*/

        Instances[] splitData = new Instances[2];
        RemoveWithValues filter = new RemoveWithValues();


        String[] options = new String[4];
        options[0] = "-C";   // Choose attribute to be used for selection
        options[1] = "1"; // Attribute number
        options[2] = "-S";   // Numeric value to be used for selection on numeric attribute. Instances with values smaller than given value will be selected. (default 0)
        options[3] = String.valueOf(splitValue);   //200. Say you want all those instances whose values for this attribute are less than 200
        filter.setOptions(options);

        filter.setInputFormat(data);
        Instances newData = Filter.useFilter(data, filter);
        System.out.println(data.numInstances());
        System.out.println(newData.numInstances());
        System.out.println(Discretize.useFilter(data, filter).numInstances());


        return splitData;
    }

    public static void main(String[] args) throws Exception {
        // Load chinaTown data from file
        FileReader reader = new FileReader("./src/main/java/ml_6002b_coursework/test_data/Chinatown.arff");
        Instances chinaTownData = new Instances(reader);
        chinaTownData.setClassIndex(chinaTownData.numAttributes()-1);

        IGAttributeSplitMeasure splitMeasure = new IGAttributeSplitMeasure();
        splitMeasure.splitDataOnNumeric(chinaTownData, chinaTownData.attribute(0), 800);

    }

}
