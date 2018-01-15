
function [theta,bestTau,MSE,R,predict] = fit_dimodel(datareal,datapred,datapredTallX,dataInd,mode,theta,options)

ffFit = @(a,datareal,datapred)sum((datareal-a(1)*datapred).^2);

switch mode
    case 'fit'
        diFit = @(a,datareal,datapred,datafakeTx)sum( (datareal-a(1)*datapred +a(2)*datafakeTx).^2) + 10*sum(a.^2);
        
        a0 = [0.1 0.05];
        times = 0:199;
        
        A = zeros(2,length(times));
        
        sse = zeros(1,length(times));
        sseAll = zeros(length(dataInd),length(times));
        
        datahat = NaN(length(times),length(dataInd));
        
        for i = 1:length(times)
            t = times(i);
            datafakeTx = datapredTallX(dataInd-t);
            A(:,i) = fminsearch(diFit, a0,options,datareal,datapred,datafakeTx);
            sse(i) = sum((datareal-A(1,i).*datapred +A(2,i).*datafakeTx).^2);
            sseAll(:,i) = (datareal-A(1,i).*datapred +A(2,i).*datafakeTx);
            datahat(i,:)  = A(1,i).*datapred -A(2,i).*datafakeTx;
        end
        [ ~,bestTaui] = min(sse);
        bestTau = times(bestTaui);
        theta(1:2) = A(1:2,bestTaui);
        theta(3) = bestTau;
        
        an = 0.2; % starting value for H0 model fit
        theta(4)= fminsearch(ffFit, an,options,datareal,datapred);
    case 'eval'
        diFitMSE = @(a,datareal,datapred,datafakeTx)sum((datareal-a(1)*datapred +a(2)*datafakeTx).^2);
        
        datafakeTx = datapredTallX(dataInd-theta(3));
        
        MSE(1) = ffFit(theta(4),datareal,datapred)/300;
        MSE(2) = diFitMSE(theta(1:2),datareal,datapred,datafakeTx)/300;
        
        R(1) = corr(theta(4)*datapred,datareal);
        R(2) = corr( theta(1)*datapred -theta(2)*datafakeTx ,datareal);
        bestTau = [];
    case 'predict'
        predict.resp = datareal;
        datafakeTx = datapredTallX(dataInd-theta(3));
        predict.DI = theta(1)*datapred -theta(2)*datafakeTx;
        predict.H0 = theta(4)*datapred;
        
        bestTau = [];
        MSE = [];
        R = [];
end