function modeloutput = di_model(data,varargin)
% function modeloutput = di_model(data,varargin)
% function fits a delayed inhibition model to motion responses.
% input: data - drifting and flashed bar responses as well as stimulus
% parameters
% inputs (optional): method - either 'default' or 'mua_composition'
%                    sigma - sigma for smoothing the firing rates (default
%                    5ms)
%                    nrBootRep - numer of repitions for bootstrapping
%                    (default 501)
%                    defaultMuaSub - mua neurons model
% output: modeloutput - simulation results
% TSH Jan 2018

defaultMethod = 'default';
defaultSigma = 5; % default 5 ms for widrh of gausssian.
defaultNrBootRep= 501;
defaultMuaSub = [0 1/20 1/10 1/5]; % only used if method 'mua_composition'

p = inputParser;
addParameter(p,'method',defaultMethod,@ischar);
addParameter(p,'sigma',defaultSigma,@isnumeric);
addParameter(p,'nrBootRep',defaultNrBootRep,@isnumeric);
addParameter(p,'muaSub',defaultMuaSub,@isnumeric);

parse(p,varargin{:});
inputs = p.Results;

sigma = inputs.sigma;
method = inputs.method;
nrBootRep = inputs.nrBootRep;
muaSub = inputs.muaSub;

%%
framerate = 100;
frametime = 1000./framerate;
flashDataTime = [-99 200];

positionMove = data.stim.positionMove;

nrSpeeds = length(data.spkMove);

if sigma>0
    nrTgauss = round(sigma*2)*2+1;
    gauss_filt = fspecial('gauss',nrTgauss,sigma);
    gauss_filt = gauss_filt(:,ceil(nrTgauss./2))./sum(gauss_filt(:,ceil(nrTgauss./2)));
else
    gauss_filt = 1;
end

centerTime = NaN(nrSpeeds,1);
for iSpeed = 1:nrSpeeds
    centerTime(iSpeed) = find(positionMove(iSpeed,:)==0,1);
end


ratesM = data.rateMove;
spkM = data.spkMove;
nrMotionTimes = size(ratesM,2);
nrCells = size(ratesM,3);
ratesF = data.rateFlash;


%%
motionTimes = repmat(0:(nrMotionTimes-1),nrSpeeds,1);
motionTimes = bsxfun(@minus,motionTimes, frametime*(centerTime-1));
motionStay = false(nrSpeeds,size(ratesM,2));
motionTimesSpk = cell(nrSpeeds,1);
motionStaySpk = cell(nrSpeeds,1);
modelStaySpk = cell(nrSpeeds,1);

modelTimes = cell(1,nrSpeeds);
timesM = cell(1,nrSpeeds);
modelStay = cell(1,nrSpeeds);

stimMove = data.stim.stimMove; % for all speeds, stimulus position over space and time
modelRawData = cell(1,nrSpeeds);
stimMoveStay = cell(1,nrSpeeds);
%%
for iSpeed = 1:nrSpeeds
    modelTimes{iSpeed} = 1:(size(stimMove{iSpeed},3));
    
    zeroMovPosition =  find(stimMove{iSpeed}(1,data.stim.positionFlash==0,:)==1,1);
    if isempty(zeroMovPosition) % 0 position wasn't shown at this speed, interpolate
        [xTmp,tTmp] = find(squeeze(stimMove{iSpeed}(1,:,:)));
        xTmp = data.stim.positionFlash(xTmp);
        zeroMovPosition = round(interp1(xTmp,tTmp',0,'linear'));
    end
    modelTimes{iSpeed} = modelTimes{iSpeed} -  zeroMovPosition ;
    modelTimes{iSpeed} =  modelTimes{iSpeed} + flashDataTime(1);
    motionStay(iSpeed,:) = motionTimes(iSpeed,:) >=  modelTimes{iSpeed}(1) & motionTimes(iSpeed,:) <= modelTimes{iSpeed}(end);
    
    timesM{iSpeed} = motionTimes(iSpeed, motionStay(iSpeed,:));
    
    modelStay{iSpeed}  = true(sum(motionStay(iSpeed,:)),1);
    modelRawData{iSpeed} = NaN(sum(motionStay(iSpeed,:)),2,nrCells);
    
    stimMoveSum = squeeze(sum(sum(stimMove{iSpeed},1),2));
    stimMoveStay{iSpeed} = stimMoveSum>0;
    
    %% now for spk
    nrSpkTimes =  size(spkM{iSpeed},1);
    motionTimesSpk{iSpeed} = (0:(nrSpkTimes-1))-frametime*(centerTime(iSpeed)-1);
    motionStaySpk{iSpeed} = motionTimesSpk{iSpeed} >=  modelTimes{iSpeed}(1) &motionTimesSpk{iSpeed} <= modelTimes{iSpeed}(end);
    modelStaySpk{iSpeed}  = true(sum(motionStaySpk{iSpeed}),1);
    
end
%%

switch method
    
    case 'default'
        
        cvModel = struct('predicion',struct,'theta',NaN,'MSE',NaN,'R',NaN, ...
            'type',method,'nrBootRep',nrBootRep,'timestart',now,'timeend',inf);
        
        pred = struct('resp',NaN,'DI',NaN,'H0',NaN);
        
        theta = NaN(4,10,nrBootRep,nrCells);
        MSE = NaN(2,10,nrBootRep,nrCells);
        R = NaN(2,10,nrBootRep,nrCells);
        options = optimset('Display','off','TolX',1.e-4,'TolFun',1.e-4,'MaxFunEvals',10000,'MaxIter',10000,'FunValCheck','off');
        
        
        h = waitbar(0,'Please wait, bootstraping in progress...');
        for iSpeed = 1:nrSpeeds
            tic
            cvModel(iSpeed).timestart = now;
            for iCell = 1:nrCells
                %%
                waitbar((iCell+(iSpeed-1)*nrCells)/(nrCells*nrSpeeds),h)
                
                rfCenterLoc = data.rfcenter(iCell);
                rf0 = (rfCenterLoc-data.stim.center_inDeg);
                
                % drifting bar response has to be aligned to rf center
                time = motionTimesSpk{iSpeed}(motionStaySpk{iSpeed});
                rfThisSpeed = 1000/data.stim.speeds(iSpeed).*rf0;
                
                timeStay = false(length(time),1);
                timeStayDelay= false(length(time)+200,1);
                [~,anaTimeStart] = min( abs(rfThisSpeed+flashDataTime(1)-time));
                timeStay(anaTimeStart:min(anaTimeStart+299,length(timeStay)),1) = true;
                
                
                timeStayDelay (max(anaTimeStart-200,1):min(anaTimeStart+299,length(timeStayDelay))) = true;
                
                if anaTimeStart-200 <1
                    addzeros = zeros(-anaTimeStart+200+1,1);
                else
                    addzeros = [];
                end
                %
                spikeFlashCell = ratesF(:,:,:,iCell);
                baseline  = mean(mean(spikeFlashCell(50:99,:)));
                spikeFlashCell = spikeFlashCell-baseline;
                spikeFlashesM1cell = repmat(spikeFlashCell,[1,1,size(stimMove{iSpeed},3)]);
                predM1cell = squeeze(nansum(nansum(spikeFlashesM1cell.*stimMove{iSpeed})));%./normbyF;
                
                predrateMthiscell = predM1cell(modelStay{iSpeed});
                datapred = conv(predrateMthiscell,gauss_filt,'same');
                datapred(~stimMoveStay{iSpeed}) = NaN;
                datapredTallX = cat(1,addzeros,datapred(timeStayDelay));
                datapred = datapred(timeStay);
                isnandatapred =  isnan(datapred);
                datapred(isnandatapred) = 0;
                dataInd =length(datapredTallX)-length(datapred)+1:length(datapredTallX);
                datapredTallX(find(isnandatapred)+200) = 0;
                nrSpeedTrials = size(spkM{iSpeed},2);
                %%
                x_train = ratesM(iSpeed,:,iCell)'-nanmean(ratesM(iSpeed,1:250,iCell));
                x_train = conv(x_train,gauss_filt,'same');
                x_train = x_train(motionStaySpk{iSpeed});
                x_train = x_train(timeStay);
                
                if any(isnan(x_train))
                    remove = isnan(x_train) | isnandatapred;
                    datapred(remove) = 0;
                    datapredTallX(find(remove)+200) = 0;
                    x_train(remove) = 0;
                else
                    x_train(isnandatapred) = 0;
                end
                
                theta_pred = fit_dimodel( x_train,datapred,datapredTallX,dataInd,'fit',[],options);
                [~,~,~,~,pred(iCell)] = fit_dimodel( x_train,datapred,datapredTallX,dataInd,'predict',theta_pred,options);
                
                %%
                
                for iBoot = 1:nrBootRep
                    %%
                    if iBoot == 1 % the first trial is actually not with random replacement but just shuffeled. Excluded later from analysis
                        bootTrials = 1000*(spkM{iSpeed}(:,randperm(nrSpeedTrials,nrSpeedTrials),iCell));
                    else
                        bootTrials = 1000*(spkM{iSpeed}(:,randi(nrSpeedTrials,nrSpeedTrials,1),iCell));
                    end
                    tenFoldSteps = ceil((0:0.1:1)*nrSpeedTrials);
                    
                    for iCross = 1:10
                        %%
                        x_test  = mean(bootTrials(:,(tenFoldSteps(iCross)+1):tenFoldSteps(iCross+1)),2);
                        x_train = mean(bootTrials(:,[1:tenFoldSteps(iCross)  (tenFoldSteps(iCross+1)+1):nrSpeedTrials] ),2);
                        
                        x_test  = x_test -nanmean(x_test(1:250)); % baseline substraction
                        x_train = x_train-nanmean(x_train(1:250));
                        
                        x_test  = conv(x_test, gauss_filt,'same');
                        x_train = conv(x_train,gauss_filt,'same');
                        
                        x_test  = x_test(motionStaySpk{iSpeed});
                        x_test  = x_test(timeStay);
                        x_train = x_train(motionStaySpk{iSpeed});
                        x_train = x_train(timeStay);
                        
                        if any(isnan(x_train))
                            remove = isnan(x_train) | isnandatapred;
                            datapred(remove) = 0;
                            datapredTallX(find(remove)+200) = 0;
                            x_train(remove) = 0;
                            x_test(remove) = 0;
                        else
                            x_train(isnandatapred) = 0;
                            x_test(isnandatapred) = 0;
                        end
                        theta(:,iCross,iBoot,iCell) = fit_dimodel( x_train,datapred,datapredTallX,dataInd,'fit',[],options);
                        [~,~,MSE(:,iCross,iBoot,iCell),R(:,iCross,iBoot,iCell)] = fit_dimodel( x_test,datapred,datapredTallX,dataInd,'eval',theta(:,iCross,iBoot,iCell));
                    end
                end
                disp([datestr(now) ': done with cell ' num2str(iCell) ' (speed ' num2str(iSpeed) ')'])
                
            end
            display([datestr(now) ': took ' num2str(toc/60) ' min for speed ' num2str(iSpeed) ' out of ' num2str(nrSpeeds)  ])
            
            cvModel(iSpeed).predicion = pred;
            cvModel(iSpeed).theta = theta;
            cvModel(iSpeed).MSE = MSE;
            cvModel(iSpeed).R = R;
            cvModel(iSpeed).timeend = now;
        end
        
        close(h)
        
        modeloutput = cvModel;
        
    case  'mua_composition'
        
        nrMuaSub = length(muaSub);
        
        muaModel = struct('prediction',struct,'theta',NaN,'MSE',NaN,'R',NaN,'type',method,'nrMuaSub',nrMuaSub,'timestart',now,'timeend',inf);
        
        theta = NaN(4,nrMuaSub,nrCells);
        MSE = NaN(2,nrMuaSub,nrCells);
        R = NaN(2,nrMuaSub,nrCells);
        pred = struct('resp',NaN,'DI',NaN,'H0',NaN);
        
        options = optimset('Display','off','TolX',1.e-4,'TolFun',1.e-4,'MaxFunEvals',10000,'MaxIter',10000,'FunValCheck','off');
        
        h = waitbar(0,'Please wait, bootstraping in progress...');
        for iSpeed = 1:nrSpeeds
            tic
            muaModel(iSpeed).timestart = now;
            for iCell = 1:nrCells
                %%
                waitbar((iCell+(iSpeed-1)*nrCells)/(nrCells*nrSpeeds),h)
                
                rfCenterLoc = data.rfcenter(iCell);
                rf0 = (rfCenterLoc-data.stim.center_inDeg);
                
                time = motionTimesSpk{iSpeed}(motionStaySpk{iSpeed});
                
                rfThisSpeed = 1000/data.stim.speeds(iSpeed).*rf0;
                
                timeStay = false(length(time),1);
                timeStayDelay= false(length(time)+200,1);
                [~,anaTimeStart] = min( abs(rfThisSpeed+flashDataTime(1)-time));
                timeStay(anaTimeStart:min(anaTimeStart+299,length(timeStay)),1) = true;
                
                timeStayDelay (max(anaTimeStart-200,1):min(anaTimeStart+299,length(timeStayDelay))) = true;
                if anaTimeStart-200 <1
                    addzeros = zeros(-anaTimeStart+200+1,1);
                else
                    addzeros = [];
                end
                
                spikeFlashCell = ratesF(:,:,:,iCell);
                baseline  = mean(mean(spikeFlashCell(50:99,:)));
                spikeFlashCell = spikeFlashCell-baseline;
                spikeFlashesM1cell = repmat(spikeFlashCell,[1,1,size(stimMove{iSpeed},3)]);
                predM1cell = squeeze(nansum(nansum(spikeFlashesM1cell.*stimMove{iSpeed})));
                
                predrateMthiscell = predM1cell(modelStay{iSpeed});
                datapred = conv(predrateMthiscell,gauss_filt,'same');
                datapred(~stimMoveStay{iSpeed}) = NaN;
                datapredTallX = cat(1,addzeros,datapred(timeStayDelay));
                datapred = datapred(timeStay);
                isnandatapred =  isnan(datapred);
                datapred(isnandatapred) = 0;
                dataInd =length(datapredTallX)-length(datapred)+1:length(datapredTallX);
                datapredTallX(find(isnandatapred)+200) = 0;
                
                
                %%
                spkMthisCell = spkM{iSpeed}(:,:,iCell);
                [t,tr] = find(spkMthisCell);
                shiftval = round(data.responseTimeDI(iSpeed,iCell));
                iShift = 1;
                
                
                
                for iSub = 1:nrMuaSub
                    %%
                    
                    spkMSub = false(size(spkMthisCell));
                    shiftT = randperm(length(t),round(length(t)*muaSub(iSub)));
                    tTmp = t;
                    tTmp(shiftT) = tTmp(shiftT) - shiftval(iShift);
                    stay = tTmp>0;
                    spkMSub(sub2ind(size(spkMSub),tTmp(stay),tr(stay))) = true;
                    
                    rateM = mean(spkMSub*1000,2);
                    
                    x_train = rateM-nanmean(rateM(1:250));
                    x_train = conv(x_train,gauss_filt,'same');
                    x_train = x_train(motionStaySpk{iSpeed});
                    x_train = x_train(timeStay);
                    
                    if any(isnan(x_train))
                        remove = isnan(x_train) | isnandatapred;
                        datapred(remove) = 0;
                        datapredTallX(find(remove)+200) = 0;
                        x_train(remove) = 0;
                    else
                        x_train(isnandatapred) = 0;
                    end
                    theta(:,iSub,iCell) = fit_dimodel( x_train,datapred,datapredTallX,dataInd,'fit',[],options);
                    
                    [~,~,~,~,pred(iSub,iCell)] = fit_dimodel( x_train,datapred,datapredTallX,dataInd,'predict',theta(:,iSub,iCell),options);
                    [~,~,MSE(:,iSub,iCell),R(:,iSub,iCell)] = fit_dimodel( x_train,datapred,datapredTallX,dataInd,'eval',theta(:,iSub,iCell));
                end
                disp([datestr(now) ': done with cell ' num2str(iCell) ' (speed ' num2str(iSpeed) ')'])
                
            end
            display([datestr(now) ': took ' num2str(toc/60) ' min for speed ' num2str(iSpeed) ' out of ' num2str(nrSpeeds)  ])
            
            muaModel(iSpeed).theta = theta;
            muaModel(iSpeed).MSE = MSE;
            muaModel(iSpeed).R = R;
            muaModel(iSpeed).prediciton = pred;
            muaModel(iSpeed).timeend = now;
        end
        
        close(h)
        
        modeloutput = muaModel;
end


end
