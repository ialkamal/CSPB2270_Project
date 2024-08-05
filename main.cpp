#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <time.h>
#include <cmath>
#include <fstream>
#include <sstream>

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
            out << "\n";

            out << "output: ";
            if(n.count("output")) out << n["output"][0] << " ";

            out << "delta: ";
            if(n.count("delta")) out << n["delta"][0] << " ";
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
    return 1.0 / (1.0 + exp(-activation));
}

vector<float> forward_propogate(vector<layer> network, vector<float> row)
{
    vector<float> inputs = row;
    float activation = -1;
    
    for(auto layer:network)
    {
        vector<float> new_inputs = {};
        for(auto neuron:layer)
        {
            activation = activate(neuron["weights"],inputs);
            neuron["output"] = vector<float>(1, transfer(activation));
            new_inputs.push_back(neuron["output"][0]);   
        }
        inputs = new_inputs;
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
    // Backpropagation loop
    for(int i=network.size()-1; i>=0; i--)
    {
        layer l = network[i];
        vector<float> errors = {};

        if(i != network.size()-1)
        {
            for(int j=0; j<l.size(); j++)
            {
                float error = 0.0;
                for(auto neuron:network[i+1])
                {
                    error += neuron["weights"][j] * neuron["delta"][0];
                }
                errors.push_back(error);
            }
        }
        else
        {
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

void read_record()
{
    fstream fin; 
    fin.open("C:\\Users\\nawatt\\Documents\\Code\\CSPB2270_Project\\wheat-seeds.csv", ios::in);
    int rollnum, roll2, count = 0; 
    vector<float> row; 
    string line, word, temp;

    

    while (fin >> temp) { 

        cout << "\nTEMP: " << temp << endl;
  
        row.clear(); 
  
        // read an entire row and 
        // store it in a string variable 'line' 
        getline(fin, line); 

        cout << "\nLINE: " << line << endl;
  
        // used for breaking words 
        stringstream s(temp); 
  
        // read every column data of a row and 
        // store it in a string variable, 'word' 
        while (getline(s, word, ',')) { 
  
            // add all the column data 
            // of a row to a vector 
            row.push_back(stof(word)); 
        } 


         for(auto a: row) cout << a << " ";
         cout << endl; 
  
        // convert string to integer for comparision 
        roll2 = int(row[0]); 
  
        // Compare the roll number 
        if (roll2 == rollnum) { 
  
            // Print the found data 
            count = 1; 
            cout << "Details of Roll " << row[0] << " : \n"; 
            cout << "Name: " << row[1] << "\n"; 
            cout << "Maths: " << row[2] << "\n"; 
            cout << "Physics: " << row[3] << "\n"; 
            cout << "Chemistry: " << row[4] << "\n"; 
            cout << "Biology: " << row[5] << "\n"; 
            break; 
        } 
    } 
    if (count == 0) 
        cout << "Record not found\n";   


}



int main()
{
    srand(time(NULL));
    vector<layer> network = initialize_network(2,1,2);
    network[0][0]["output"] = {0.7105668883115941};
    network[1][0]["output"] = {0.6213859615555266};
    network[1][1]["output"] = {0.6573693455986976};
    
    cout << network;

    vector<float> row = {1, 0};

    vector<float> output = forward_propogate(network, row);

    vector<float> expected = {0,1};

    backward_propogate_error(network, expected);
    cout << network;


    cout<<"Hello World!";

    read_record();

    return 0;
}