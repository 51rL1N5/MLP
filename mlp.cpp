#define ARMA_DONT_USE_WRAPPER
#include <iostream>
#include <armadillo>
#include <vector>
#include <functional>
#include <cmath>

#define ETA   0.1
#define ALPHA 0.1


using namespace std;
using namespace arma;



int main(int argc, char** argv)
{
  int N_H_LAYERS     =              1;
  int N_TOTAL_LAYERS = N_H_LAYERS + 2;

  int n_neurons[] = { 2 ,2, 1};

  Mat <double> W[N_TOTAL_LAYERS - 1];


  function<void (Mat<double> & )> sigma   = [] (Mat<double> & M)
                                         {
                                           M.for_each([](double & val)
                                           {
                                             val = 1 / ( 1+ exp(-val));
                                           }
                                       );};
  function<void (Mat<double> &)> d_sigma = [] (Mat<double> & M)
                                         {
                                           M.for_each([](double & val)
                                           {
                                             val = (1 / ( 1+ exp(-val))) * ( 1 - 1 / ( 1+ exp(-val)) ) ;
                                           }
                                       );};


  // GERAR AS MATRIZES DE PESO ALEATORIOS
  // DE DIMENSAO (N_NEURONS_CAMADA_ATUAL + 1 (BIAS))x (N_NEURONS_PROXIMA_CAMADA)
  for (int i = 0; i < N_TOTAL_LAYERS - 1; i++)
    W[i].randu(n_neurons[i+1], n_neurons[i] + 1);


  // PROPAGACAO DIRETA

  // VETOR DE MATRIZES Y (SAIDAS DE CADA CAMADA COM A ATIVACAO)
  Mat<double> Y[N_TOTAL_LAYERS];

  // Y[0] CORRESPONDE A ENTRADA
  Y[0] = {
          {0,0, 1, 1},
          {0,1, 0, 1},
         };

  // UNITARY BIAS
  Mat<double> B(1,4);
  B(0,0) = -1;
  B(0,1) = -1;
  B(0,2) = -1;
  B(0,3) = -1;

  for (int i = 1; i <= N_TOTAL_LAYERS; i++)
  {
    cout << W[i-1] << endl;
    cout << Y[i-1] << endl;
    Y[i] =  W[i-1] * join_cols( B, Y[i-1]);
    sigma( Y[i] );
  }

  cout << Y[N_TOTAL_LAYERS];
  return 0;
}
