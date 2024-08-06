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

using namespace std;
typedef map<string,vector<float>> neuron;
typedef vector<neuron> layer;


vector<float> generate_weights(int input)
{
    vector<float> list;
    
    for(int i=0; i<input+1;i++)
    {
        float random_number = rand() / float(RAND_MAX);
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

float activate(vector<float> weights, vector<float> inputs)
{
    float activation = weights.back();
    for(int i=0; i<weights.size()-1; i++)
    {
        activation += weights[i] * inputs[i];
    }
    return activation;
}

float transfer(float activation)
{
    return 1.0 / (1.0 + exp(-1*activation));
}

vector<float> forward_propogate(vector<layer>& network, vector<float> row)
{
    vector<float> inputs = row;
    float activation = -1;
    
    for(int i=0; i<network.size();i++)
    {
        vector<float> new_inputs = {};
        for(int j=0; j<network[i].size(); j++)
        {
            activation = activate(network[i][j]["weights"],inputs);
            network[i][j]["output"] = vector<float>(1, transfer(activation));
            new_inputs.push_back(network[i][j]["output"][0]);   
        }
        inputs = {new_inputs.begin(),new_inputs.end()};
    }

    return inputs;
}

float transfer_derivative(float output)
{
    return output * (1.0 - output);
}

void backward_propogate_error(vector<layer>& network, vector<float> expected){

    neuron n;
    // Initialize delta for each neuron in each layer
    for (int i = 0; i < network.size(); i++) {
        for (int j = 0; j < network[i].size(); j++) {
            network[i][j]["delta"] = vector<float>(1, 0.0); 
        }
    }

    // cout << "BB BEFORE" << endl;
    // cout << network << endl;

    // Backpropagation loop
    for(int i=network.size()-1; i>=0; i--)
    {
        layer l = network[i];
        vector<float> errors = {};

        if(i != network.size()-1)
        {

            // cout << endl << "NEXT!!!!" << endl;

            for(int j=0; j<l.size(); j++)
            {
                float error = 0.0;
                for(auto neuron:network[i+1])
                {
                    error += neuron["weights"][j] * neuron["delta"][0];
                    // cout << endl << "ERRORR FOUND!!!!" << endl;
                }
                errors.push_back(error);
            }
        }
        else
        {
            // cout << endl << "REACHED!!!!" << endl;

            for(int j=0; j<l.size(); j++)
            {
                n = l[j];
                errors.push_back(n["output"][0] - expected[j]);
            }
        }

        for(int j=0; j<l.size(); j++)
        {
            network[i][j]["delta"][0] = errors[j] * transfer_derivative(n["output"][0]);
        }
    }
}

void update_weights(vector<layer>& network, vector<float> row, float l_rate)
{
    vector<float> inputs;
    
    for(int i = 0; i<network.size(); i++)
    {

        if(i != 0)
        {
            inputs.clear();
            for(auto neuron:network[i-1])
                inputs.push_back(neuron["output"][0]);
        }
        else
        {
            inputs = {row.begin(),row.end()-1};
            
        }


        // cout << "VECTOR: " << endl;
        // for(auto inp:inputs) cout << inp << " ";
        // cout << endl;

        // cout << endl << "SIZE: " << inputs.size() << endl;
        cout << endl << "ROW: " << endl;
        for(auto a:inputs) cout << a << " ";
        cout << endl;

        for(int j = 0; j<network[i].size(); j++)
        {
            for(int k=0; k < inputs.size(); k++)
                network[i][j]["weights"][k] -= l_rate * network[i][j]["delta"][0] * inputs[k];
                
        
            network[i][j]["weights"][network[i][j]["weights"].size()-1] -= l_rate * network[i][j]["delta"][0];
        }

    }
}

void train_network(vector<layer>& network, vector<vector<float>> train, float l_rate, int n_epoch, int n_outputs)
{
    

    for(int epoch=0; epoch<n_epoch; epoch++)
    {
        float sum_error = 0;

        for(auto row:train)
        {
            vector<float> outputs = forward_propogate(network, row);

            // cout << "Outputs: " << endl;
            // for(auto output:outputs) cout << output << " ";
            // cout << endl;
            
            vector<float> expected = vector<float>(n_outputs, 0);

            // cout << "Expected: " << endl;
            // for(auto e:expected) cout << e << " ";
            // cout << endl;

            expected[row[row.size()-1]] = 1;

            // cout << "Expected (After): " << endl;
            // for(auto e:expected) cout << e << " ";
            // cout << endl;

            for(int i=0; i<expected.size(); i++)
                sum_error += pow(expected[i] - outputs[i], 2);

            // cout << "Sum Error: " << sum_error << endl;

            backward_propogate_error(network, expected); 

            // cout << "Back Propagation: " << endl;
            // cout << network;

            update_weights(network, row, l_rate);

            // cout << "Update Weights: " << endl;
            // cout << network;           

        }
        cout << "epoch= " << epoch << " lrate= " << l_rate << " error= " << sum_error << endl;
    }
}

int predict(vector<layer>& network, vector<float> row)
{
    vector<float> outputs = forward_propogate(network, row);
    cout << " " << *max_element(outputs.begin(),outputs.end()) << endl;
    return max_element(outputs.begin(),outputs.end()) - outputs.begin();
}

vector<vector<float>> read_record()
{
    fstream fin; 
    fin.open("../wheat-seeds.csv", ios::in);
    int rollnum, roll2, count = 0; 
    vector<vector<float>> record;
    vector<float> row; 
    string line, word, temp;

    

    while (fin >> temp) { 

        row.clear(); 
  
        // read an entire row and 
        // store it in a string variable 'line' 
        // getline(fin, line); 

        // cout << "\nLINE: " << line << endl;
  
        // used for breaking words 
        stringstream s(temp); 
  
        // read every column data of a row and 
        // store it in a string variable, 'word' 
        while (getline(s, word, ',')) { 
  
            // add all the column data 
            // of a row to a vector 
            row.push_back(stof(word)); 
        }


        record.push_back(row);
    } 
    

    // for(auto line:record)
    // {
    //     for(auto val:line)
    //     {
    //         cout << val << ",";
    //     }
    //     cout << endl;
    // }
    
    if (record.size() == 0) 
        cout << "Record not found\n";   

    return record;

}



int main()
{
    srand(time(NULL));

    cout << endl;
    cout<<"Begin Neural Network Program!" << endl;

    // vector<vector<float>> dataset = { {2.7810836 ,2.550537003, 0 },
    //                                   {1.465489372, 2.362125076, 0},
    //                                   {3.396561688, 4.400293529, 0},
    //                                   {1.38807019, 1.850220317, 0},
    //                                   {3.06407232, 3.005305973, 0},
    //                                   {7.627531214, 2.759262235, 1},
    //                                   {5.332441248, 2.088626775, 1},
    //                                   {6.922596716, 1.77106367, 1},
    //                                   {8.675418651, -0.242068655, 1},
    //                                   {7.673756466, 3.508563011, 1} };

    vector<vector<float>> dataset = read_record();

    int n_inputs = dataset[0].size() - 1;
    
    set<int> classes;
    for(auto row:dataset)
    {
        classes.insert(row[row.size()-1]);
    }
    int n_outputs = classes.size();

    vector<layer> network = initialize_network(n_inputs,3,n_outputs);

    cout << "Inputs= " << n_inputs << " Outputs= " << n_outputs << endl;

    cout << endl << "Network Structure (Before): " << endl;
    cout << network;

    train_network(network, dataset, 0.3, 20, n_outputs);
    
    cout << endl << "Network Structure (After): " << endl;
    cout << network << endl;

    for(auto row:dataset)
    {
        int prediction = predict(network, row);
        cout << "Expected = " << row[row.size()-1] << " , Predicted = " << prediction << endl;
    } 
    
    
    // network[0][0]["output"] = {0.7105668883115941};
    // network[1][0]["output"] = {0.6213859615555266};
    // network[1][1]["output"] = {0.6573693455986976};
    
    // cout << network;

    // vector<float> row = {1, 0};

    // vector<float> output = forward_propogate(network, row);

    // vector<float> expected = {0,1};

    // cout << endl << "Back Propagation:" << endl;
    // backward_propogate_error(network, expected);
    // cout << network;


    // cout<<"Hello World!";

    
    return 0;
}