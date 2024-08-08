#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <time.h>
#include <cmath>
#include <fstream>
#include <sstream>
#include <set>
#include <algorithm>
#include <iterator>
#include <iomanip>   


using namespace std;
typedef map<string,vector<double>> neuron;
typedef vector<neuron> layer;


vector<double> generate_weights(int input)
{
    vector<double> list;
    
    for(int i=0; i<input+1;i++)
    {
        double random_number = rand() / double(RAND_MAX);
        list.push_back(random_number);
    }
    return list;  
}

vector<layer> initialize_network(int n_inputs, int n_hidden, int n_outputs){

    vector<layer> network;
    
    vector<neuron> hidden_layer;
    for(int i = 0; i<n_hidden; i++)
    {
        neuron n;
        n["weights"] = generate_weights(n_inputs);
        hidden_layer.push_back(n);
    }

    network.push_back(hidden_layer);

    vector<neuron> output_layer;
    for(int i = 0; i<n_outputs; i++)
    {
        neuron n;
        n["weights"] = generate_weights(n_hidden);
        output_layer.push_back(n);
    }   

    network.push_back(output_layer);

    return network;
 
}

std::ostream& operator<<(std::ostream& out, const vector<layer>& e) {

        out << "\n";
    
        for(auto layer:e){
            for(auto n:layer)
        {
            out << "weights: ";
            for(auto num:n["weights"]) {
                out << num << " ";
            }

            out << "output: ";
            if(n.count("output")) out << n["output"][0] << " ";

            out << "delta: ";
            if(n.count("delta")) out << n["delta"][0] << " ";

            out<<"\n";
        }
        out<<"\n";
    }
    
    return out;
}

double activate(vector<double> weights, vector<double> inputs)
{
    double activation = weights.back();
    for(int i=0; i<weights.size()-1; i++)
    {
        activation += weights[i] * inputs[i];
    }
    return activation;
}

double transfer(double activation)
{
    return 1.0 / (1.0 + exp(-activation));
}

vector<double> forward_propogate(vector<layer>& network, vector<double> row)
{
    vector<double> inputs = row;
    double activation = -1;
    
    for(int i=0; i<network.size();i++)
    {
        vector<double> new_inputs = {};
        for(int j=0; j<network[i].size(); j++)
        {
            activation = activate(network[i][j]["weights"],inputs);
            //cout << "ACtivation: " << activation << endl;
            network[i][j]["output"] = vector<double>(1, transfer(activation));
            new_inputs.push_back(network[i][j]["output"][0]);   
        }
        inputs = new_inputs;
    }

    return inputs;
}

double transfer_derivative(double output)
{
    return output * (1.0 - output);
}

void backward_propogate_error(vector<layer>& network, vector<double> expected){

    // Initialize delta for each neuron in each layer
    for (int i = 0; i < network.size(); i++) {
        for (int j = 0; j < network[i].size(); j++) {
            network[i][j]["delta"] = vector<double>(1, 0.0); 
        }
    }

    // Backpropagation loop
    for(int i=network.size()-1; i>=0; i--)
    {
        //vector<double> errors = {};

        for(int j=0; j<network[i].size(); j++)
        {
            double error = 0.0;
            if(i == network.size()-1)
             {
                error = expected[j] - network[i][j]["output"][0];
             }
            else
             {
                for(auto neuron:network[i+1])
                    error += neuron["weights"][j] * neuron["delta"][0];
             }
            
            network[i][j]["delta"][0] = error * transfer_derivative(network[i][j]["output"][0]);

        }      
    }
    
}

void update_weights(vector<layer>& network, vector<double> row, double l_rate)
{
    
    
    for(int i = 0; i<network.size(); i++)
    {
        vector<double> inputs = {};

        if(i != 0)
        {
            for(auto neuron:network[i-1])
                inputs.push_back(neuron["output"][0]);
        }
        else
            inputs = {row.begin(),row.end()-1};

        for(int j = 0; j<network[i].size(); j++)
        {
            for(int k=0; k < inputs.size(); k++)
                network[i][j]["weights"][k] += l_rate * network[i][j]["delta"][0] * inputs[k];
                
        
            network[i][j]["weights"][network[i][j]["weights"].size()-1] += l_rate * network[i][j]["delta"][0];
        }

    }
}

void train_network(vector<layer>& network, vector<vector<double>> train, double l_rate, int n_epoch, int n_outputs)
{
    

    for(int epoch=0; epoch<n_epoch; epoch++)
    {
        double sum_error = 0.0;

        for(auto row:train)
        {
            vector<double> outputs = forward_propogate(network, row);
            
            vector<double> expected = vector<double>(n_outputs, 0);
            expected[row.back()] = 1;

            for(int i=0; i<n_outputs; i++)
                sum_error += pow(expected[i] - outputs[i], 2);

            backward_propogate_error(network, expected); 


            update_weights(network, row, l_rate);

        }
        cout << "epoch= " << epoch << " lrate= " << l_rate << " error= " << sum_error << endl;
    }
}

int predict(vector<layer>& network, vector<double> row)
{
    vector<double> outputs = forward_propogate(network, row);
    return max_element(outputs.begin(),outputs.end()) - outputs.begin();
}

vector<vector<double>> read_record()
{
    fstream fin; 
    fin.open("../wheat-seeds.csv", ios::in);
    vector<vector<double>> record;
    vector<double> row; 
    string word, temp;
    

    while (fin >> temp) { 

        double min, max;
        row.clear(); 
        stringstream s(temp); 
  
        // read every column data of a row and 
        // store it in a string variable, 'word' 
        while (getline(s, word, ',')) { 
  
            // add all the column data 
            // of a row to a vector
            row.push_back(stof(word)); 
        }

        //Normalize Data
        min = *min_element(row.begin(), row.end()-1);
        max = *max_element(row.begin(), row.end()-1);
        for(int i=0; i < row.size()-1; i++)
            row[i] = (row[i]- min) / (max - min);

        record.push_back(row);
    } 
    
    if (record.size() == 0) 
        cout << "Record not found\n";   

        

    return record;

}

double accuracy_metric(vector<int> actual, vector<int> predicted)
{
    double correct = 0;
    for(int i=0; i<actual.size(); i++)
        if(actual[i] == predicted[i]) correct++;

    return (correct / double(actual.size())) * 100.0;
}

vector<vector<vector<double>>> cross_validation_split(vector<vector<double>> dataset, int n_folds)
{
    vector<vector<vector<double>>> dataset_split = {};
    int fold_size = int(dataset.size() / n_folds);

    vector<vector<double>> dataset_copy = {dataset.begin(), dataset.end()};

    for(int i=0; i<n_folds; i++)
    {
        vector<vector<double>> fold = {};
        while(fold.size() < fold_size)
        {
            int index = rand() % dataset_copy.size();
            fold.push_back(dataset_copy.at(index));
            dataset_copy.erase(dataset_copy.begin() + index);
        }
        dataset_split.push_back(fold);
    }
    return dataset_split;
}

vector<double> evaluate_algorithm(vector<vector<double>> dataset, int n_folds)
{
   vector<vector<vector<double>>> folds =  cross_validation_split(dataset, n_folds);
   vector<double> scores = {};


   for(int i=0; i<n_folds; i++)
   {

    cout << endl << "Cross Validation: Fold No. " << i << endl; 

    vector<vector<double>> train = {};
    train.reserve(folds[0].size()*(n_folds-1));
    vector<vector<double>> test = {};
    for(int j=0; j<n_folds; j++)
    {
        if(i==j)
            test = {folds[j].begin(), folds[j].end()};
        train.insert(train.end(),folds[j].begin(),folds[j].end());
    } 

    int n_inputs = dataset[0].size() - 1;
    
    set<int> classes;
    for(auto row:dataset)
    {
        classes.insert(row[row.size()-1]);
    }
    int n_outputs = classes.size();

    vector<layer> network = initialize_network(n_inputs,5,n_outputs);

    train_network(network, train, .3, 500, n_outputs);

    vector<int> predictions = {};
    vector<int> actual = {};
    for(auto row:test)
    {
        actual.push_back(row.back());
        int prediction = predict(network, row);
        predictions.push_back(prediction);
    } 

    scores.push_back(accuracy_metric(actual,predictions));
   }
   
    return scores;
}


int main()
{
    srand(time(NULL));

    cout << endl;
    cout<<"Begin Neural Network Program!" << endl;

    vector<vector<double>> dataset = read_record();
    vector<double> scores = evaluate_algorithm(dataset, 5);

    double mean_score = 0;
    for(auto score:scores) mean_score += score / scores.size();
    cout << endl << "======RESULTS======" << endl;
    cout << "Mean Accuracy of Neural Network Algorithm: " << setprecision(4) << mean_score << "%" << endl;
    cout << "===================";
    cout<<endl;

    return 0;
}