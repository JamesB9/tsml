package ml_6002b_coursework;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.io.FileReader;


public class ChiSquaredAttributeSplitMeasure extends AttributeSplitMeasure {

    @Override
    public double computeAttributeQuality(Instances data, Attribute att) throws Exception {
        int[][] contingencyTable = new int[att.numValues()][data.numClasses()];

        for(Instance instance : data){
            contingencyTable[(int) instance.value(att)][(int) instance.classValue()]++;
        }

        return AttributeMeasures.measureChiSquared(contingencyTable);
    }

    /**
     * Main method.
     *
     * @param args the options for the split measure main
     */
    public static void main(String[] args) throws Exception {

        // Load Whisky data from file
        FileReader reader = new FileReader("./src/main/java/ml_6002b_coursework/test_data/WhiskyRegion.arff");
        Instances whiskeyData = new Instances(reader);
        whiskeyData.setClassIndex(whiskeyData.numAttributes()-1);

        // Create Split Measure
        ChiSquaredAttributeSplitMeasure splitMeasure = new ChiSquaredAttributeSplitMeasure();

        // Measure Chi Squared
        for(int attr = 0; attr < whiskeyData.numAttributes() - 1; attr++){
            double infoGain = splitMeasure.computeAttributeQuality(whiskeyData, whiskeyData.attribute(attr));
            String attributeName = whiskeyData.attribute(attr).name();
            System.out.printf("measure 'Chi Squared' for attribute '%s' splitting diagnosis = %f%n",
                    attributeName, infoGain);
        }
    }

}
