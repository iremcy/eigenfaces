load('Yale_32x32.mat');
basis_vector = [0,27,43,59,70,80,90,95];
for file = 2:8
    total_nn = 0;
    total_nc = 0;
    for number = 1:50
        nn_score = 0;
        nc_score = 0;
        load(strcat(num2str(file),'Train\',num2str(number),'.mat'));
        
        fea_Train = fea(trainIdx,:);
        fea_Test = fea(testIdx,:);
        gnd_Train = gnd(trainIdx);
        gnd_Test = gnd(testIdx);

        [train_size,y] = size(fea_Train);
        [test_size,y] = size(fea_Test);

        % FEA_TRAIN COMPRESSING
        nu = mean(fea_Train);
        data = fea_Train - repmat(nu,train_size,1);
        covariance = cov(data);
        [EigVec,EigVal] = eig(covariance); %it returns a diagonal eigenvalue matrix
        EigVal = diag(EigVal); %descending
        EigVec = flipud(EigVec); %ascending
        FeatureVector = EigVec(:,1:basis_vector(file)); %largest eigenvalues
        FinalData = FeatureVector' * data';

        % FEA_TEST COMPRESSION
        data = fea_Test - repmat(nu,test_size,1);
        FinalData_Test = FeatureVector' * data';
        
        % NN CLASSIFICATION
        X=FinalData_Test';
        Y=FinalData';
        % find the nearest neighbors for each test data
        % compare the results with gnd_Test
        % calculate the accuracy
        classify = knnsearch(Y,X,'NSMethod','euclidian');
        % for each number in classify, take the gnd_Train value
        for i = 1:test_size
            classify(i) = gnd_Train(classify(i));
            if(classify(i) == gnd_Test(i))
                nn_score = nn_score + 1;
            end
        end
        
        % NC CLASSIFICATION
        % find the mean of each class
        Data = FinalData';
        j = 0;
        class_label = 0;
        centroid = [];
        mem_num = 0;
        for i = 1:train_size
            if(class_label ~= gnd_Train(i))% new class
                if(j~=0)
                    centroid(j,:) = centroid(j,:)/mem_num;
                    j = j +1;
                    centroid(j,:) = Data(i,:);
                    mem_num = 1;
                end
                if(j == 0)
                    j = j +1;
                    centroid(j,:) = Data(i,:);
                    mem_num = 1;
                end
            end
            if(class_label == gnd_Train(i))
                mem_num = mem_num +1;
                centroid(j,:) = centroid(j,:) + Data(i,:);
            end
            class_label = gnd_Train(i);
        end        
        % find the closest centroid
        X=FinalData_Test';
        classify = knnsearch(centroid,X,'NSMethod','euclidian');
        for i = 1:test_size
            if(classify(i) == gnd_Test(i))
                nc_score = nc_score + 1;
            end
        end
        total_nn = total_nn + ((test_size - nn_score)*100)/test_size;
        total_nc = total_nc + ((test_size - nc_score)*100)/test_size;
    end
    display(strcat('File ',num2str(file),': nn: ',num2str(total_nn/50),' nc: ',num2str(total_nc/50)));
end