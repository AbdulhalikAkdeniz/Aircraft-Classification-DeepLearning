%%Only feature extraction
clc, clear all, close all

dosyalar = dir('airCraftDataset/*.*g');

netA = vgg16;
layer = "fc8";
inputSizeA = netA.Layers(1).InputSize; %224x224x3

% Parça boyutu
patchSize = 112; %224/2

X = []; % resim özellik matrisi
y = []; % etiket matrisi

sayac=1;
for i = 1:length(dosyalar) %veri hazırlığı
    res = imread(fullfile("airCraftDataset", dosyalar(i).name));
    res = imresize(res, inputSizeA(1:2)); 

    augimdsTrainA_full = augmentedImageDatastore(inputSizeA(1:2), res); % tam resim veri artırma
    featuresTrain_full = activations(netA, augimdsTrainA_full, layer, 'OutputAs', 'rows'); % tam resim özellik çıkarımı

    X(i, 1:size(featuresTrain_full, 2)) = featuresTrain_full; % tam resim özelliklerin saklanması
    
    % Dört parçaya bölme döngüsü
    for ii = 1:2  % İki parça yatayda
        for jj = 1:2  % İki parça dikeyde
            % Parçayı al
            patch = res(1 + (ii-1)*patchSize:ii*patchSize, 1 + (jj-1)*patchSize:jj*patchSize, :);

            % veri artırma işlemi
            augimdsTrainA_patch = augmentedImageDatastore(inputSizeA(1:2), patch);

            % özellik çıkarma
            featuresTrain_patch  = activations(netA, augimdsTrainA_patch, layer, 'OutputAs', 'rows');

            startCol = (sayac) * size(featuresTrain_patch, 2) + 1;
            endCol = startCol + size(featuresTrain_patch, 2) - 1;
            X(i, startCol:endCol) = featuresTrain_patch;

            sayac=sayac+1;
        end
    end
    %y(i, 1) = str2num(dosyalar(i).name(1)); % her görüntüye sınıf etiketi ataması yaptık
    [sayi, ~] = strtok(dosyalar(i).name, ' '); 
    y(i, 1) = str2double(sayi);
    sayac=1;
end

%%
%After feature extraction

Xx = X;
Xx=(Xx-min(Xx))./(max(Xx)-min(Xx)+eps);

coeff_X = pca(Xx);
X_pca = Xx * coeff_X(:, 1:1000);
son = horzcat(X_pca,y);

ldaModel = fitcdiscr(X_pca, y);

%%
%Test
testResim = imread('test.jpg');
testResim = imresize(testResim, inputSizeA(1:2)); % Eğitimde kullanılan boyuta getir
augimdsTest = augmentedImageDatastore(inputSizeA(1:2), testResim);
featuresTest = activations(netA, augimdsTest, layer, 'OutputAs', 'rows');
sayacTest=1;
    % Dört parçaya bölme döngüsü
    for ii = 1:2  % İki parça yatayda
        for jj = 1:2  % İki parça dikeyde
            patchTest = testResim(1 + (ii-1)*patchSize:ii*patchSize, 1 + (jj-1)*patchSize:jj*patchSize, :);
            augimdsTest_patch = augmentedImageDatastore(inputSizeA(1:2), patchTest);
            featuresTest_patch  = activations(netA, augimdsTest_patch, layer, 'OutputAs', 'rows');
            startCol = (sayacTest) * size(featuresTest_patch, 2) + 1;
            endCol = startCol + size(featuresTest_patch, 2) - 1;
            featuresTest(1, startCol:endCol) = featuresTest_patch;
            sayacTest=sayacTest+1;
        end
    end
testVerii = featuresTest;
testVerii=(testVerii-min(testVerii))./(max(testVerii)-min(testVerii)+eps);

pcaFeaturesTest = testVerii * coeff_X(:, 1:1000);

tahmin = predict(ldaModel, pcaFeaturesTest);
etiketler = ["A-10", "B-52", "E-3", "F-22", "KC-10", "B-1", "B-2", "B-29", "Boeing", "C-130", "C-135", "C-17", "C-5", "F-16", "C-21", "U-2", "A-26", "P-63", "T-6", "T-43"];
tahminString = etiketler(tahmin);
disp('Test Görüntüsünün Tahmin Edilen Etiketi: '+ tahminString);

%Tekil tahmin sonu

%%
%toplu test
tstKlasoru = 'TestAirCraft';
tstDosyalar = dir(fullfile(tstKlasoru, '*.jpg'));

tahminler_tst = [];
gercekEtiketler_tst = [];

for it = 1:length(tstDosyalar)
    tstResim = imread(fullfile(tstKlasoru, tstDosyalar(it).name));
    tstResim = imresize(tstResim, inputSizeA(1:2)); % Eğitimde kullanılan boyuta getir
    augimdsTst = augmentedImageDatastore(inputSizeA(1:2), tstResim);
    featuresTst = activations(netA, augimdsTst, layer, 'OutputAs', 'rows');
    sayacTst=1;
        % Dört parçaya bölme döngüsü
        for ii = 1:2  % İki parça yatayda
            for jj = 1:2  % İki parça dikeyde
                patchTst = tstResim(1 + (ii-1)*patchSize:ii*patchSize, 1 + (jj-1)*patchSize:jj*patchSize, :);
                augimdsTst_patch = augmentedImageDatastore(inputSizeA(1:2), patchTst);
                featuresTst_patch  = activations(netA, augimdsTst_patch, layer, 'OutputAs', 'rows');
                startCol_T = (sayacTst) * size(featuresTst_patch, 2) + 1;
                endCol_T = startCol_T + size(featuresTst_patch, 2) - 1;
                featuresTst(1, startCol_T:endCol_T) = featuresTst_patch;
                sayacTst=sayacTst+1;
            end
        end
    %testVerii = featuresTest;
    featuresTst=(featuresTst-min(featuresTst))./(max(featuresTst)-min(featuresTst)+eps);
    pcaFeaturesTst = featuresTst * coeff_X(:, 1:1000);
    % Modeli kullanarak tahminde bulun
    tahmin_tst = predict(ldaModel, pcaFeaturesTst);
    tahminler_tst(it,1) = tahmin_tst;
    gercekEtiketler_tst(it,1) = str2num(tstDosyalar(it).name(1));
    sayacTst=1;
end
dogruTahminSayisi = sum(tahminler_tst == gercekEtiketler_tst);
% Doğruluk oranını hesapla
dogrulukOrani = (dogruTahminSayisi / length(tstDosyalar)) * 100;
fprintf('Test veri seti için Doğruluk Oranı: %.2f%%\n', dogrulukOrani);
tahminKiyas_tst = horzcat(gercekEtiketler_tst,tahminler_tst);

%Toplu test sonu
