package ml_6002b_coursework;

import weka.classifiers.AbstractClassifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Random;

public class TreeEnsemble extends AbstractClassifier {

    private int numTrees = 50;
    private CourseworkTree[] trees = new CourseworkTree[numTrees];
    private ArrayList<Attribute>[] treeAttributes = new ArrayList[numTrees];
    private double attributeProportionPerTree = 0.5;
    private boolean averageDistributions = false;

    TreeEnsemble() {};

    TreeEnsemble(int numTrees) {
        numTrees = numTrees;
        trees = new CourseworkTree[numTrees];
        treeAttributes = new ArrayList[numTrees];
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        // Initialise the treeAttributes store
        Random random = new Random();
        // Build Random Subspace
        for(int i = 0; i < numTrees; i++){
            // Create and build tree
            trees[i] = new CourseworkTree();

            trees[i].buildClassifier(data);

            // Select random splitting measure
            String[] options = new String[2];
            options[0] = "-M";
            switch(random.nextInt(4)){
                case 0:
                    options[1] = "informationGain";
                    break;
                case 1:
                    options[1] = "informationGainRatio";
                    break;
                case 2:
                    options[1] = "chiSquared";
                    break;
                case 3:
                    options[1] = "gini";
                    break;
            }

            // Select random attributes
            int numberOfAttributes = (int) (attributeProportionPerTree * data.numAttributes());
            // Add all attributes to the subset
            ArrayList<Attribute> attributesSubset = new ArrayList<>();
            for(int att = 0; att < data.numAttributes(); att++){
                attributesSubset.add(data.attribute(att));
            }
            // Randomly remove attributes until left with desired amount
            while(attributesSubset.size() > numberOfAttributes){
                attributesSubset.remove(random.nextInt(attributesSubset.size()));
            }
            treeAttributes[i] = attributesSubset;
        }
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        int[] votes = new int[instance.numClasses()];
        int winningClass = 0;

        if(averageDistributions){
            // Finds the highest average predicted class
            for (int tree = 0; tree < numTrees; tree++) {
                // Delete attributes that tree doesn't use
                for(int attribute = 0; attribute < instance.numAttributes(); attribute++){
                    //instance.deleteAttributeAt();
                }

                double[] predictions = trees[tree].distributionForInstance(instance);
                for(int classVotes = 0; classVotes < votes.length; classVotes++){
                    votes[classVotes] += predictions[classVotes];
                    if (votes[classVotes] > votes[winningClass]) winningClass = classVotes;
                }
            }
        }else {
            // Count votes and keep track of class with majority votes
            for (int tree = 0; tree < numTrees; tree++) {
                int prediction = (int) trees[tree].classifyInstance(instance);
                votes[prediction]++;
                if (votes[prediction] > votes[winningClass]) winningClass = prediction;
            }

        }
        return winningClass;
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        double[] votes = new double[instance.numClasses()];

        // Count Votes
        for(int tree = 0; tree < numTrees; tree++){
            int prediction = (int) trees[tree].classifyInstance(instance);
            votes[prediction]++;
        }
        // Convert vote counts to the proportion of votes for each class
        for(int classVotes = 0; classVotes < votes.length; classVotes++){
            votes[classVotes] = votes[classVotes] / (double) numTrees;
        }

        return votes;
    }

    @Override
    public void setOptions(String[] options) throws Exception {
        super.setOptions(options);
        switch (Utils.getOption('A', options).toLowerCase()){
            case "true":
                averageDistributions = true;
                break;
            case "false":
                averageDistributions = false;
                break;
        }
    }

    public static void main(String[] args) throws Exception {
        String[] dataFiles = {
                "./src/main/java/ml_6002b_coursework/test_data/optdigits.arff",
                "./src/main/java/ml_6002b_coursework/test_data/Chinatown.arff"
        };

        for(String dataFile : dataFiles){
            // Load data
            FileReader reader = new FileReader(dataFile);
            Instances data = new Instances(reader);
            data.setClassIndex(data.numAttributes()-1);

            // Randomly split data in between training and test data sets

            data.randomize(new java.util.Random(0)); // Randomize (System.currentTimeMillis() for randomness)
            int trainAmount = (int) (0.5f * data.numInstances()); // 70% train data
            int testAmount = data.numInstances() - trainAmount; // 30% test data
            Instances trainData = new Instances(data, 0, trainAmount);
            Instances testData = new Instances(data, trainAmount, testAmount);

            // Create and Build Classifiers

            // Information Gain Tree
            TreeEnsemble treeEnsemble = new TreeEnsemble();
            treeEnsemble.buildClassifier(trainData);

            int correct = 0;

            for(int i = 0; i < testData.numInstances(); i++){
                Instance instance = testData.instance(i);
                double prediction = treeEnsemble.classifyInstance(instance);

                if(i < 5){ // the probability estimates for the first five test cases
                    double[] distribution = treeEnsemble.distributionForInstance(instance);
                    System.out.printf("TreeEnsemble test case %d on %s problem has probability estimates = %s\n", i,
                            data.relationName(),Arrays.toString(distribution));
                }

                if(prediction == instance.classValue()) correct++;
            }
            double accuracy =  (double)correct / (double)testData.numInstances();

            System.out.printf("TreeEnsemble on %s problem has test accuracy = %f\n", data.relationName(), accuracy*100);
        }
    }
}
