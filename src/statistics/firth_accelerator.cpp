// Numerical acceleration of the archived Firth algorithm, not a new estimator.
#include <cmath>
#include <algorithm>
#include <random>
#include <cstdint>
#include <omp.h>

#if defined(_WIN32) || defined(__CYGWIN__)
#  define RVPL_EXPORT __declspec(dllexport)
#else
#  define RVPL_EXPORT __attribute__((visibility("default")))
#endif

constexpr int P=8, N=64;
double clamp(double x,double lo,double hi){return std::max(lo,std::min(hi,x));}
bool invert(double a[P][P],int p,double inv[P][P],double &ld){
 double L[P][P]={}; ld=0;
 for(int i=0;i<p;i++)for(int j=0;j<=i;j++){
  double v=a[i][j];for(int k=0;k<j;k++)v-=L[i][k]*L[j][k];
  if(i==j){if(v<=1e-15)return false; L[i][j]=std::sqrt(v);ld+=2*std::log(L[i][j]);}
  else L[i][j]=v/L[j][j];
 }
 for(int c=0;c<p;c++){
  double z[P]={},u[P]={};
  for(int i=0;i<p;i++){double v=i==c?1.:0.;for(int k=0;k<i;k++)v-=L[i][k]*z[k];z[i]=v/L[i][i];}
  for(int i=p-1;i>=0;i--){double v=z[i];for(int k=i+1;k<p;k++)v-=L[k][i]*u[k];u[i]=v/L[i][i];}
  for(int i=0;i<p;i++)inv[i][c]=u[i];
 }
 return true;
}
double evaluate(const double*X,const double*y,const double*b,int n,int p,double*pr,double*w,double info[P][P]){
 for(int j=0;j<p;j++)for(int k=0;k<p;k++)info[j][k]=0;
 double ll=0.;
 for(int i=0;i<n;i++){
  double eta=0.;for(int j=0;j<p;j++)eta+=X[i*p+j]*b[j];eta=clamp(eta,-35,35);
  pr[i]=1/(1+std::exp(-eta));w[i]=std::max(pr[i]*(1-pr[i]),1e-9);
  ll+=y[i]*std::log(std::max(pr[i],1e-12))+(1-y[i])*std::log(std::max(1-pr[i],1e-12));
  for(int j=0;j<p;j++)for(int k=0;k<p;k++)info[j][k]+=w[i]*X[i*p+j]*X[i*p+k];
 }
 double inv[P][P],ld;
 return invert(info,p,inv,ld)?ll+.5*ld:-INFINITY;
}
int fit(const double*X,const double*y,int n,int p,double*b,double*se){
 for(int j=0;j<p;j++)b[j]=0;
 double pr[N],w[N],A[P][P],inv[P][P],ld;int converged=0;
 for(int it=0;it<300;it++){
  double old=evaluate(X,y,b,n,p,pr,w,A);
  if(!invert(A,p,inv,ld))return -1;
  double score[P]={},step[P]={};
  for(int i=0;i<n;i++){
   double h=0.;for(int j=0;j<p;j++)for(int k=0;k<p;k++)h+=w[i]*X[i*p+j]*inv[j][k]*X[i*p+k];
   for(int j=0;j<p;j++)score[j]+=X[i*p+j]*(y[i]-pr[i]+h*(.5-pr[i]));
  }
  for(int j=0;j<p;j++)for(int k=0;k<p;k++)step[j]+=inv[j][k]*score[k];
  double scale=1.,nb[P];
  while(true){
   for(int j=0;j<p;j++)nb[j]=b[j]+scale*step[j];
   if(scale<=1e-7||evaluate(X,y,nb,n,p,pr,w,A)>=old)break;
   scale*=.5;
  }
  double diff=0.;for(int j=0;j<p;j++){diff=std::max(diff,std::abs(nb[j]-b[j]));b[j]=nb[j];}
  if(diff<1e-8){converged=1;break;}
 }
 evaluate(X,y,b,n,p,pr,w,A);if(!invert(A,p,inv,ld))return -1;
 for(int j=0;j<p;j++)se[j]=std::sqrt(std::max(inv[j][j],0.));return converged;
}
extern "C" RVPL_EXPORT void fit_one(double*X,double*y,int n,int p,double*b,double*s,int*ok){*ok=fit(X,y,n,p,b,s);}
extern "C" RVPL_EXPORT void permute(double*Z,double*fitted,double*r,double*y,int n,int p,int B,uint64_t seed,double*out,int*status){
 omp_set_num_threads(10);
 #pragma omp parallel for schedule(dynamic,32)
 for(int k=0;k<B;k++){
  std::mt19937_64 rng(seed+uint64_t(k)*0x9e3779b97f4a7c15ULL);int idx[N];for(int i=0;i<n;i++)idx[i]=i;
  std::shuffle(idx,idx+n,rng);double X[N*P],b[P],s[P];
  for(int i=0;i<n;i++){for(int j=0;j<p;j++)X[i*(p+1)+j]=Z[i*p+j];X[i*(p+1)+p]=fitted[i]+r[idx[i]];}
  status[k]=fit(X,y,n,p+1,b,s);out[k]=b[p];
 }
}
extern "C" RVPL_EXPORT void calibrate(double*Z,double*fitted,double*r,double*ys,int n,int p,int S,int B,uint64_t seed,double*out){
 omp_set_num_threads(10);
 #pragma omp parallel for schedule(dynamic,1)
 for(int sim=0;sim<S;sim++){
  const double*y=ys+sim*n;double X[N*P],b[P],s[P];
  for(int i=0;i<n;i++){for(int j=0;j<p;j++)X[i*(p+1)+j]=Z[i*p+j];X[i*(p+1)+p]=fitted[i]+r[i];}
  int obsok=fit(X,y,n,p+1,b,s);double obs=b[p],se=s[p];int extreme=0,fail=0;
  std::mt19937_64 rng(seed+uint64_t(sim)*0x9e3779b97f4a7c15ULL);
  for(int k=0;k<B;k++){
   int idx[N];for(int i=0;i<n;i++)idx[i]=i;std::shuffle(idx,idx+n,rng);
   for(int i=0;i<n;i++)X[i*(p+1)+p]=fitted[i]+r[idx[i]];
   int ok=fit(X,y,n,p+1,b,s);if(ok!=1)fail++;
   if(std::abs(b[p])>=std::abs(obs))extreme++;
  }
  out[sim*5]=(extreme+1.)/(B+1.);out[sim*5+1]=obs;out[sim*5+2]=se;out[sim*5+3]=fail;out[sim*5+4]=obsok;
 }
}
