package ml_6002b_coursework;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.NumericToBinary;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.instance.RemoveWithValues;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.Arrays;
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
    public Instances[] splitData(Instances data, Attribute att) throws Exception {

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


    // TODO:
    // Sort data
    // Iterate through data, splitting at each
    // For each split, calculate the Sum of Squares Error (SSE)
    // Variable with least SSE is chosen as split

    // Sources:
    // Breiman, Leo, et al. Classification and regression trees. CRC press, 1984.
    //https://stats.stackexchange.com/questions/220350/regression-trees-how-are-splits-decided
    // https://gdcoder.com/decision-tree-regressor-explained-in-depth/
    // https://medium.com/analytics-vidhya/regression-trees-decision-tree-for-regression-machine-learning-e4d7525d8047
    public Instances[] splitDataOnNumeric(Instances data, Attribute att) throws Exception {
        GiniAttributeSplitMeasure giniSplitMeasure = new GiniAttributeSplitMeasure();

        // Sort Data into ascending order
        data.sort(att);

        // Set the attribute from numeric type to nominal
        NumericToNominal numericToNominal = new NumericToNominal();
        numericToNominal.setInputFormat(data);
        numericToNominal.setAttributeIndicesArray(new int[]{att.index()});

        double bestSplitValue = 0.0;
        double minGini = 1000000.0;

        for(Instance instance1 : data){
            double splitValue = instance1.value(att);
            // Create copy of data
            Instances nominalData = new Instances(data);

            // Sets the attribute values to 0 or 1 depending upon splitValue
            for(Instance instance2 : nominalData) {
                if(instance2.value(att) <= splitValue) instance2.setValue(att, 0);
                else instance2.setValue(att, 1);
            }

            // Filter nominal data
            numericToNominal.setInputFormat(nominalData);
            nominalData = Filter.useFilter(nominalData, numericToNominal);

            // GINI See if better split
            double gini = giniSplitMeasure.computeAttributeQuality(nominalData, nominalData.attribute(att.index()));

            if(gini < minGini && gini > 0.0001) {
                bestSplitValue = instance1.value(att);
                minGini = gini;
            }
        }

        Instances[] splitData = new Instances[2];
        splitData[0] = new Instances(data, data.numInstances());
        splitData[1] = new Instances(data, data.numInstances());

        for (Instance instance : data) {
            if(instance.value(att) <= bestSplitValue) splitData[0].add(instance);
            else splitData[1].add(instance);
        }

        splitData[0].compactify();
        splitData[1].compactify();

        return splitData;
    }

    public static void main(String[] args) throws Exception {
        // Load chinaTown data from file
        FileReader reader = new FileReader("./src/main/java/ml_6002b_coursework/test_data/Chinatown.arff");
        Instances chinaTownData = new Instances(reader);
        chinaTownData.setClassIndex(chinaTownData.numAttributes()-1);

        IGAttributeSplitMeasure splitMeasure = new IGAttributeSplitMeasure();
        System.out.println(splitMeasure.computeAttributeQuality(chinaTownData, chinaTownData.attribute(0)));

    }

}
