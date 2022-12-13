% x is the observations size dimension x n
function [df, mu, sigma] = mle_MVT(x,rho)

    %%%%%%%%    k mu1 mu2 mu3 s11 s12 s13 s22 s23 s33
    initvec = [ 4 1   1   1   1.5 -1  0   2   1   1.5];
    tol=1e-5;
    opts=optimset('Disp','none','LargeScale','Off','TolFun',tol, ...
        'TolX ',tol,'Maxiter',200);
    % minimum of negative log likelihood
    % maximize loglikelihood
    mle =  fminunc(@(param) tloglik(param,x,rho),initvec,opts);
    
    df = mle(1);
    mu = mle(2:4);
    sigma = zeros(3,3);
    sigma(1,:) = mle(5:7);
    sigma(2,1) = sigma(1,2);
    sigma(3,1) = sigma(1,3);
    sigma(2,2) = mle(8);
    sigma(2,3) = mle(9);
    sigma(3,2) = sigma(2,3);
    sigma(3,3) = mle(10);
end

function ll = tloglik(param,x,rho)
    % param is k mu1 mu2 mu3 s11 s12 s13 s22 s23 s33
    df = param(1);
    mu = param(2:4);
    sigma = zeros(3,3);
    sigma(1,:) = param(5:7);
    sigma(2,1) = sigma(1,2);
    sigma(3,1) = sigma(1,3);
    sigma(2,2) = param(8);
    sigma(2,3) = param(9);
    sigma(3,2) = sigma(2,3);
    sigma(3,3) = param(10);
    
    % make T dynamic for x
    T = 2000;
    tvec=(1:T); 
    omega=(T-tvec+1).^(rho-1); 
    w=T*omega'/sum(omega);

    if min(eig(sigma))<1e-10
        ll=1e5;
    else 
        z = x.'-mu;
        pdf = mvtpdf(z,sigma,df);
        llvec = log(pdf).*w;
        ll = -sum(llvec);
    end
end