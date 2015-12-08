function [alfa, a, b] = build_robMPM_lin_binclass_LSreg(mX,mY,covX,covY,nu,rho_x,rho_y,gauss_assump,algoparam,tol,maxiter);
% build_robMPM_lin_binclass_LSreg - build a robust linear minimax probability machine (MPM) for binary classification
%                                   using iterative least squares with regularization on the systemmatrix LSmat 
%                                   (LSmat is an internal variablle, i.e., systemmatrix for least squares step)
%
% [alfa, a, b] = build_robMPM_lin_binclass_LSreg(mX,mY,covX,covY,nu,rho_x,rho_y,gauss_assump,algoparam,tol,maxiter)
%
%
% The algorithm finds the robust minimax probabilistic decision hyperplane between two classes of 
% points x and y
%
% H = {z | a'*z = b}
%
% that maximizes alfa (lower bound on the probability of correct classification of future data) subject
% to the constraint a<>0 and
%
% inf_(x~DX) Pr(a'x >= b) >= alfa
% inf_(y~DY) Pr(a'y <= b) >= alfa
%
% where the infimum is taken over DX, resp. DY, being the set of all distributions for x, resp. y, having
% a mean and covariance matrix in the convex set V, resp. W (if gauss_assump=1, only Gaussian distributions are 
% considered):
%
% V = {mean(x),cov(x) : (mean(x)-mX)^T inv(cov(x)) (mean(x)-mX) <= nu^2,
%                        ||cov(x)-covX||_F <= rho_x}
% W = {mean(y),cov(y) : (mean(y)-mY)^T inv(cov(y)) (mean(y)-mY) <= nu^2,
%                        ||cov(y)-covY||_F <= rho_y}
% 
% where mX and covX, resp. mY and covY, are the estimates for the mean and covariance matrix of class x, resp. class y, 
% provided in the input; ||...||_F denotes the Frobenius norm.
%
%
% More details can be found in
%
% Lanckriet, G.R.G., El Ghaoui, L., Bhattacharyya, C., Jordan, M.I. (2002). A Robust Minimax Approach
% to Classification. Journal of Machine Learning Research. 
% URL: http://robotics.eecs.berkeley.edu/~gert/index.htm
%
%
%
% The inputs are
% mX           - estimated mean of class x (column vector)
% mY           - estimated mean of class y (column vector)
% covX         - estimated covariance matrix of class x
% covY         - estimated covariance matrix of class y
% nu            - robustness parameter quantizing uncertainty in the mean of both classes
%                 (0 for no robustness)
% rho_x,rho_y   - robustness parameter quantizing uncertainty in the covariance for class x, resp. y
%                 (0 for no robustness)
% gauss_assump  - 1 if x and y are assumed to be Gaussian distributed / 0 if not (this will only
%                 influence the optimal value of alfa, not the position of the optimal hyperplane; more details
%                 about this can be found in the reference)
% algoparam     - internal parameter to determine the amount of regularization added to LSmat, the systemmatrix
%                 for the least squares step; technically, algopar * eye(size(LSmat)) is added to LSmat
%                 enter -1 to use default value: 1.000000e-006
% tol           - relative tolerance level for least squares iterations
%                 enter -1 to use default value: 1.000000e-006
% maxiter       - maximum number of iterations for least squares iterations
%                 enter -1 to use  default value: 50
%
% The outputs are
% alfa         - lower bound on the probability of correct classification of future data
% a, b         - model parameters for the linear MPM
%
%
% Gert Lanckriet, October 2002.


%%%%%%%%%%%%% INITIALIZATION STEPS %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%% set default values if needed %%%%%%
if algoparam==-1
    algoparam=1e-06;
end
if tol==-1
    tol=1e-06;
end
if maxiter==-1
    maxiter=50;
end

%%%%%% build matrices needed for iterative least squares (see Section 2.3 in reference) %%%%%%
% vector a0
d = (mX - mY);
a0 = d/(d'*d);
% matrix F -- orthogonal matrix whose columns span the subspace of vectors orthogonal to a0
N = length(a0);
f = zeros(1,N-1);
[maxel,maxind]=max(abs(d));
for i=1:maxind-1,
   f(1,i) = -d(i,1)/(maxel*sign(d(maxind)));
end
for i=maxind:N-1,
   f(1,i) = -d(i+1,1)/(maxel*sign(d(maxind)));
end
IN_1 = eye(N-1);
F = [IN_1(1:maxind-1,:); f; IN_1(maxind:N-1,:)];
% matrices for least squares step
% reg_covX = covX + rho_x*eye(N);
% reg_covY = covY + rho_y*eye(N);

% F'*a0

if isscalar(rho_x)
    reg_covX = covX + rho_x*diag(diag(covX)); % !!!!!!!modified by Gao
    reg_covY = covY + rho_y*diag(diag(covY));
elseif isvector(rho_x)
    reg_covX = covX + diag(rho_x);
    reg_covY = covY + diag(rho_y);
else
    reg_covX = covX + rho_x; 
    reg_covY = covY + rho_y;
end

% G = F'*reg_covX*F;
% H = F'*reg_covY*F;
g = F'*(reg_covX*a0);
h = F'*(reg_covY*a0);

ind_tmp=[true(maxind-1,1);false;true(N-maxind,1)];
reg_covX13=reg_covX(ind_tmp,ind_tmp);
reg_covX1232=reg_covX(ind_tmp,maxind);
reg_covX22=reg_covX(maxind,maxind);
reg_covY13=reg_covY(ind_tmp,ind_tmp);
reg_covY1232=reg_covY(ind_tmp,maxind);
reg_covY22=reg_covY(maxind,maxind);

G=reg_covX13+reg_covX1232*f+f'*reg_covX1232'+reg_covX22*f'*f;
H=reg_covY13+reg_covY1232*f+f'*reg_covY1232'+reg_covY22*f'*f;
% g=reg_covX(ind_tmp,:)*a0+f'*(reg_covX(maxind,:)*a0);
% h=reg_covY(ind_tmp,:)*a0+f'*(reg_covY(maxind,:)*a0);


%%%%%%%%%%%%% ITERATIVE LEAST SQUARES %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%% (see Section 2.3 in reference) %%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%% initialization %%%%%%
beta_k = 1;
eta_k = 1;
iter = 1;
rel_obj_ch = 10*tol;
rel_beta_ch = 10*tol;
rel_eta_ch = 10*tol;
u_k = 0;

%%%%%% iterations %%%%%%%
while and(rel_obj_ch > tol,iter<maxiter),
   LSmat = G/beta_k + H/eta_k;
   LSmat = LSmat + algoparam*eye(size(LSmat));
   LSvect = -(g/beta_k + h/eta_k);
   u_k = LSmat\LSvect;
   a_k = a0 + F*u_k;
   
   
   arg1 = a_k'*reg_covX*a_k;
   arg2 = a_k'*reg_covY*a_k;
   beta_kp1 = sqrt(arg1);
   eta_kp1 = sqrt(arg2);
   obj_old = 2*(beta_k + eta_k);
   obj_new = 2*(beta_kp1 + eta_kp1);
   rel_obj_ch = abs(obj_new-obj_old)/abs(obj_old);
   iter = iter + 1;
%    disp(iter)
   beta_k = beta_kp1;
   eta_k = eta_kp1;
end


%%%%%%%%%%%%% ASSIGN OUTPUTS %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

a = a_k;
%%%%% should be the same
b = mean([a'*mX - (beta_k/(beta_k+eta_k)) a'*mY + (eta_k/(beta_k+eta_k))]);
kappa = 1/(beta_k+eta_k);
mm = max([0 kappa-nu]);
% mm=abs(mX'*a-mY'*a)/(sqrt(a'*covX*a)+sqrt(a'*covY*a)); % modified for semi-MPM
if gauss_assump==1
    alfa = normcdf(mm,0,1);
else
    alfa = mm^2/(1+mm^2);
end