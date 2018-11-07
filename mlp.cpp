#define ARMA_DONT_USE_WRAPPER
#include <iostream>
#include <armadillo>
#include <vector>
#include <functional>
#include <cmath>
#include <ctime>
#include <cstdlib>

#define ETA   0.2
#define ALPHA 0


using namespace std;
using namespace arma;



int main(int argc, char** argv)
{

  srand(time(NULL));

  int N_H_LAYERS     =              1;
  int N_TOTAL_LAYERS = N_H_LAYERS + 2;

  int n_neurons[] = { 2 ,2, 1};

  Mat <double> W[N_TOTAL_LAYERS - 1];

  int epoch = 0;

  Mat <double> d = {-1, 1, 1, -1};
  Mat <double> error;
  Mat<double> delta;
  double MSE;

  function<Mat<double> (Mat<double> )> sigma   = [] (Mat<double> M)
                                               {
                                                 M.for_each([](double & val)
                                                 {
                                                   val = 1 / ( 1+ exp(-val));
                                                 }
                                               );
                                                return M;
                                               };
  function<Mat<double> (Mat<double> )> d_sigma = [] (Mat<double> M)
                                               {
                                                 M.for_each([](double & val)
                                                 {
                                                   val = (1 / ( 1+ exp(-val))) * ( 1 - 1 / ( 1+ exp(-val)) ) ;
                                                 }
                                               );
                                                 return M;
                                                };


  // GERAR AS MATRIZES DE PESO ALEATORIOS
  // DE DIMENSAO (N_NEURONS_CAMADA_ATUAL + 1 (BIAS))x (N_NEURONS_PROXIMA_CAMADA)
  for (int i = 0; i < N_TOTAL_LAYERS - 1; i++)
    W[i].randu(n_neurons[i+1], n_neurons[i] + 1);


  // PROPAGACAO DIRETA

  // VETOR DE MATRIZES Y (SAIDAS DE CADA CAMADA COM A ATIVACAO)
  Mat<double> Y[N_TOTAL_LAYERS], dW[N_TOTAL_LAYERS - 1] ;

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

  double lim = pow(10,-30);

  do
  {

  for (int i = 1; i < N_TOTAL_LAYERS; i++)
  {
    Y[i] =  W[i-1] * join_cols( B, Y[i-1]);
    Y[i] = sigma( Y[i] );
  }


  error = d - Y[N_TOTAL_LAYERS -1];

  MSE = trace(error.t() * error)/4;
  if ( MSE <= lim || epoch == 10000) break;

  for (int i = N_TOTAL_LAYERS - 2; i > 0; i--)
  {
  //  dW[N_TOTAL_LAYERS - 2] = ETA * delta * join_cols( B, Y[i-1]) + ALPHA * dW[N_TOTAL_LAYERS - 2];
  //  W[N_TOTAL_LAYERS - 2] = W[N_TOTAL_LAYERS - 2];

    if (i == N_TOTAL_LAYERS - 2)
      delta = d_sigma( Y[i+1]) % error;
    else
      delta = d_sigma(join_cols( B, Y[i+1])) % (W[i+1].t() * delta);

    if (epoch != 0)
      dW[i] = ETA * delta * join_cols(B,Y[i]).t() + ALPHA * dW[i];
    else
      dW[i] = ETA * delta* join_cols(B,Y[i]).t();
/*
    cout << "------------------------" << endl;
    cout << Y[i + 1] << endl << endl;
    cout << delta << endl << endl;
    cout << dW[i] << endl << endl;
    cout << W[i] << endl;
*/
    W[i] = W[i] + dW[i];
  }
  epoch++;

  cout <<"Epoch: " << epoch <<" MSE: " << MSE <<" BEST : " << Y[N_TOTAL_LAYERS - 1] << "\n------------------" << endl;
  }while(true);

  cout << Y[N_TOTAL_LAYERS - 1];
  return 0;
}
