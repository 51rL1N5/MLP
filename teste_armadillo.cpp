#define ARMA_DONT_USE_WRAPPER
#include <iostream>
#include <armadillo>

using namespace std;
using namespace arma;

int main(int argc, char** argv)
{

  Mat<double> D(2,2);

  for (int i = 0; i < D.n_rows; i++)
  {
    for (int j = 0; j < D.n_cols; j++)
    {
      D(i,j) = i + j;
    }
  }

  Mat< double> On = ones(1,2);

  On = - On;

  D = join_cols(On,D);

  cout << D;

  return 0;
}
