package ml_6002b_coursework;

/**
 * Empty class for Part 2.1 of the coursework.
 */
public class AttributeMeasures {

    private static double log2(double x){
        if(x == 0) {
            return 0;
        }else {
            return Math.log(x) / Math.log(2);
        }
    }


    public static double measureInformationGain(int[][] contingencyTable) {
        int classCount = contingencyTable[0].length;
        int attrValueCount = contingencyTable.length;

        double[] entropyList = new double[attrValueCount];
        double[] weightList = new double[attrValueCount];
        int totalInRootNode = 0;

        // Calculate total in root node
        int[] rootClassTotals = new int[classCount];
        for (int[] attr : contingencyTable) {
            for (int outcome = 0; outcome < classCount; outcome++) {
                rootClassTotals[outcome] += attr[outcome];
                totalInRootNode += attr[outcome];
            }
        }

        for(int node = 0; node < attrValueCount; node++) {
            // Find total elements in node
            int totalInNode = 0;
            for (int classAmount : contingencyTable[node]) {
                totalInNode += classAmount;
            }

            // Calculate Probabilities of each class
            double[] probList = new double[classCount];
            for(int outcome = 0; outcome < classCount; outcome++){
                probList[outcome] = (double) contingencyTable[node][outcome] / (double) totalInNode;
            }

            // Calculate Entropy of each node
            double entropy = 0;
            for (double prob : probList) {
                entropy += log2(prob) * prob;
            }
            entropyList[node] = -1 * entropy;

            // Calculate Weight of each node
            weightList[node] = (double) totalInNode / (double) totalInRootNode;
        }


        // Calculate Probabilities of each class of root node
        double[] probList = new double[classCount];
        for(int outcome = 0; outcome < classCount; outcome++){
            probList[outcome] = (double) rootClassTotals[outcome] / (double) totalInRootNode;
        }
        // Calculate Entropy of root node
        double rootNodeEntropy = 0;
        for (double prob : probList) {
            rootNodeEntropy += log2(prob) * prob;
        }
        rootNodeEntropy *= -1;



        // Calculate IG
        double informationGain = rootNodeEntropy;
        for(int node = 0; node < contingencyTable.length; node++) {
            informationGain -= weightList[node] * entropyList[node];
        }
        return informationGain;
    }

    public static double measureInformationGainRatio(int[][] contingencyTable) {
        double informationGain = measureInformationGain(contingencyTable);
        double splitInfo = 0;

        int classCount = contingencyTable[0].length;
        int attrValueCount = contingencyTable.length;
        double[] weightList = new double[attrValueCount];
        int totalInRootNode = 0;

        // Calculate total in root node
        for (int[] attr : contingencyTable) {
            for (int outcome = 0; outcome < classCount; outcome++) {
                totalInRootNode += attr[outcome];
            }
        }

        // Calculate weights of each node
        for(int node = 0; node < attrValueCount; node++) {
            // Find total elements in node
            int totalInNode = 0;
            for (int classAmount : contingencyTable[node]) {
                totalInNode += classAmount;
            }
            // Calculate Weight of each node
            weightList[node] = (double) totalInNode / (double) totalInRootNode;
        }

        // Calculate Split Info
        for(double weight : weightList){
            splitInfo += weight * log2(weight);
        }
        splitInfo *= -1;

        double informationGainRatio = informationGain / splitInfo;
        return informationGainRatio;
    }

    public static double measureGini(int[][] contingencyTable) {
        int classCount = contingencyTable[0].length;
        int attrValueCount = contingencyTable.length;

        double rootImpurity = 0.0;
        int totalInRootNode = 0;
        double[] impurityList = new double[attrValueCount];


        // Calculate total in root node
        int[] rootClassTotals = new int[classCount];
        for (int[] attr : contingencyTable) {
            for (int outcome = 0; outcome < classCount; outcome++) {
                rootClassTotals[outcome] += attr[outcome];
                totalInRootNode += attr[outcome];
            }
        }

        // Calculate root node impurity
        for(int count = 0; count < classCount; count++){
            double fraction = (double) rootClassTotals[count] / (double) totalInRootNode;
            rootImpurity += Math.pow(fraction, 2);
        }
        rootImpurity = 1 - rootImpurity;


        // Calculate Gini for each node
        for(int node = 0; node < attrValueCount; node++) {
            // Find total elements in node
            int totalInNode = 0;
            for (int classAmount : contingencyTable[node]) {
                totalInNode += classAmount;
            }

            // Calculate  impurity
            double impurity = 0.0;
            for(int count = 0; count < classCount; count++){
                double fraction = (double) contingencyTable[node][count] / (double) totalInNode;
                impurity += Math.pow(fraction, 2);
            }
            impurity = 1 - impurity;
            impurityList[node] = impurity;
        }

        double gini = rootImpurity;
        for(int node = 0; node < attrValueCount; node++) {
            // Find total elements in node
            int totalInNode = 0;
            for (int classAmount : contingencyTable[node]) {
                totalInNode += classAmount;
            }

            gini -= ((double) totalInNode / (double) totalInRootNode) * impurityList[node];
        }

        return gini;
    }

    public static double measureChiSquared(int[][] contingencyTable) {
        int classCount = contingencyTable[0].length;
        int attrValueCount = contingencyTable.length;

        double chiSquared = 0.0;
        int totalInRootNode = 0;

        // Calculate total in root node
        int[] rootClassTotals = new int[classCount];
        for (int[] attr : contingencyTable) {
            for (int outcome = 0; outcome < classCount; outcome++) {
                rootClassTotals[outcome] += attr[outcome];
                totalInRootNode += attr[outcome];
            }
        }

        // Calculate Probabilities of each class of root node
        double[] probList = new double[classCount];
        for(int outcome = 0; outcome < classCount; outcome++){
            probList[outcome] = (double) rootClassTotals[outcome] / (double) totalInRootNode;
        }


        for(int node = 0; node < attrValueCount; node++) {
            // Find total elements in node
            int totalInNode = 0;
            for (int classAmount : contingencyTable[node]) {
                totalInNode += classAmount;
            }

            for(int outcome = 0; outcome < classCount; outcome++){
                double actualValue = contingencyTable[node][outcome];
                double expectedValue = probList[outcome] * totalInNode;
                chiSquared += Math.pow(actualValue - expectedValue, 2) / expectedValue;
            }
        }

        return chiSquared;
    }

    /**
     * Main method.
     *
     * @param args the options for the attribute measure main
     */
    public static void main(String[] args) {
        int[][] peatyContingencyTable = {
                {4, 0}, // Yes (4 - islay, 0 - speyside)
                {1, 5}  // No  (1 - islay, 5 - speyside)
        };
        double informationGain = measureInformationGain(peatyContingencyTable);
        double informationGainRatio = measureInformationGainRatio(peatyContingencyTable);
        double chiSquared = measureChiSquared(peatyContingencyTable);
        double gini = measureGini(peatyContingencyTable);
        System.out.printf("measure information gain for peaty = %f%n", informationGain);
        System.out.printf("measure information gain ratio for peaty = %f%n", informationGainRatio);
        System.out.printf("measure chi squared for peaty = %f%n", chiSquared);
        System.out.printf("measure gini for peaty = %f%n", gini);
    }

}
