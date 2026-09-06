// Stub replacement for BOLT-LMM_v2.5/src/NonlinearOptMulti.cpp.
//
// The upstream file is the only consumer of NLopt in BOLT-LMM. It implements the
// REML-AI (average-information) path, which is *not* reached under --lmmInfOnly.
// Replacing it with this throwing stub lets the reference build link without
// NLopt installed, while still failing loudly if the path is ever exercised.
#include "NonlinearOptMulti.hpp"
#include <stdexcept>

namespace NonlinearOptMulti {
  namespace ublas = boost::numeric::ublas;

  std::vector < ublas::matrix <double> > constrainedNR
  (double &dLLpred, ublas::vector <double> &p,
   const std::vector < ublas::matrix <double> > &Vegs,
   const ublas::vector <double> &grad,
   const ublas::matrix <double> &AI,
   double maxStepNorm) {
    throw std::runtime_error("NLopt-dependent REML-AI path is disabled in the lmmInfOnly reference build");
  }
}
