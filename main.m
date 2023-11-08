clc
clear
load data2.mat; %этот файл создается calibration.m
mdl = 'rlSimplePendulumModel2'; %здесь выбирается какой файл симулинка подгрузить
agent = [mdl '/RL Agent'];%в файле симулинка должен быть блок RL Agent
obsInfo = rlNumericSpec([3 1],'LowerLimit',-inf,'UpperLimit',inf); %здесь указывается количество входящих сигналов и управляющих (то есть 3 и 1).
obsInfo.Name = 'observations';
Ts = 0.05;%частота управления
Tf = 20;%длительность работы
theta0=0;
actInfo = rlNumericSpec([1 1],'LowerLimit',-2,'UpperLimit',2); %здесь можем указать максимальное и минимальное управляющее воздействие (указано -2 и 2)
actInfo.Name = 'torque';
%env = rlPredefinedEnv("SimplePendulumModel-Continuous");%в оригинале все строки выше отсутствовали и вместо них была эта
env = rlSimulinkEnv(mdl,agent,obsInfo,actInfo);%манипуляции выше были нужны для того, чтобы создать этот блок данных environments

%Далее оригинальные строки, в которых создавались параметры обучения
set_param( ...
    "rlSimplePendulumModel2/create observations", ...
    "ThetaObservationHandling","sincos");
%env.ResetFcn = @(in)setVariable(in,"theta0",pi,"Workspace",mdl);

rng(0)
doTraining = true;
% Define state path
statePath = [
    featureInputLayer( ...
        obsInfo.Dimension(1), ...
        Name="obsPathInputLayer")
    fullyConnectedLayer(400)
    reluLayer
    fullyConnectedLayer(300,Name="spOutLayer")
    ];

% Define action path
actionPath = [
    featureInputLayer( ...
        actInfo.Dimension(1), ...
        Name="actPathInputLayer")
    fullyConnectedLayer(300, ...
        Name="apOutLayer", ...
        BiasLearnRateFactor=0)
    ];

% Define common path
commonPath = [
    additionLayer(2,Name="add")
    reluLayer
    fullyConnectedLayer(1)
    ];

% Create layergraph, add layers and connect them
criticNetwork = layerGraph();
criticNetwork = addLayers(criticNetwork,statePath);
criticNetwork = addLayers(criticNetwork,actionPath);
criticNetwork = addLayers(criticNetwork,commonPath);
criticNetwork = connectLayers(criticNetwork,"spOutLayer","add/in1");
criticNetwork = connectLayers(criticNetwork,"apOutLayer","add/in2");
criticNetwork = dlnetwork(criticNetwork);
summary(criticNetwork)
%plot(criticNetwork)
critic = rlQValueFunction(criticNetwork, ...
    obsInfo,actInfo, ...
    ObservationInputNames="obsPathInputLayer", ...
    ActionInputNames="actPathInputLayer");
actorNetwork = [
    featureInputLayer(obsInfo.Dimension(1))
    fullyConnectedLayer(400)
    reluLayer
    fullyConnectedLayer(300)
    reluLayer
    fullyConnectedLayer(1)
    tanhLayer
    scalingLayer(Scale=max(actInfo.UpperLimit))
    ];
actorNetwork = dlnetwork(actorNetwork);
summary(actorNetwork)
actor = rlContinuousDeterministicActor(actorNetwork,obsInfo,actInfo);
criticOpts = rlOptimizerOptions(LearnRate=1e-03,GradientThreshold=1);
actorOpts = rlOptimizerOptions(LearnRate=1e-04,GradientThreshold=1);
agentOpts = rlDDPGAgentOptions(...
    SampleTime=Ts,...
    CriticOptimizerOptions=criticOpts,...
    ActorOptimizerOptions=actorOpts,...
    ExperienceBufferLength=1e6,...
    DiscountFactor=0.99,...
    MiniBatchSize=128);
agentOpts.NoiseOptions.Variance = 0.6;
agentOpts.NoiseOptions.VarianceDecayRate = 1e-5;
agent = rlDDPGAgent(actor,critic,agentOpts);
maxepisodes = 5000;
maxsteps = ceil(Tf/Ts);
trainOpts = rlTrainingOptions(...
    MaxEpisodes=maxepisodes,...
    MaxStepsPerEpisode=maxsteps,...
    ScoreAveragingWindowLength=5,...
    Verbose=false,...
    Plots="training-progress",...
    StopTrainingCriteria="AverageReward",...
    StopTrainingValue=2500,...
    SaveAgentCriteria="EpisodeReward",...
    SaveAgentValue=2500);

if doTraining
    % Train the agent.
    trainingStats = train(agent,env,trainOpts);
else
    % Load the pretrained agent for the example.
   % load("SimulinkPendulumDDPG.mat","agent")
end
simOptions = rlSimulationOptions(MaxSteps=500);
experience = sim(env,agent,simOptions);
