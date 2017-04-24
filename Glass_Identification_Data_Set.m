%%-------------------------------------------------------------------------
%%Project- EE503 Naive Bayes Classifier
%%To Simulate Naive Bayes Classifier and finding the Joint and Marginal Density function
%%-------------------------------------------------------------------------
%%Data set: Glass Identification Data Set 
%%https://archive.ics.uci.edu/ml/datasets/Glass+Identification
%%-------------------------------------------------------------------------
%%Author                       Date                Revision
%%Harish & Rajasekar Raja     04/07/17         Initial Revision
%%-------------------------------------------------------------------------
clear all;
clc;
%Input training and testing data being read from a .txt file
training_data = csvread('Glass_Train.txt');
training_data_size = size(training_data,1);
test_data = csvread('Glass_Test.txt');
test_data_size  = size(test_data,1);
fprintf('\nTraining the Gaussian Naive Bayes Classifier with Training Data\n');

%Separating all the Training samples into separate classes
train_class1=[]; train_class2=[]; train_class3=[]; train_class4=[]; train_class5=[]; train_class6=[]; train_class7=[];
for iter =1:training_data_size
    if training_data(iter,11) == 1
        train_class1 = cat(1,train_class1,training_data(iter,2:10));
    elseif training_data(iter,11) == 2
        train_class2 = cat(1,train_class2,training_data(iter,2:10));
    elseif training_data(iter,11) == 3
        train_class3 = cat(1,train_class3,training_data(iter,2:10));
    elseif training_data(iter,11) == 4
        train_class4 = cat(1,train_class4,training_data(iter,2:10));    
    elseif training_data(iter,11) == 5
        train_class5 = cat(1,train_class5,training_data(iter,2:10));
    elseif training_data(iter,11) == 6
        train_class6 = cat(1,train_class6,training_data(iter,2:10));
    elseif training_data(iter,11) == 7
        train_class7 = cat(1,train_class7,training_data(iter,2:10));
    end
end
%Separating all the Test samples into separate classes
test_class1=[]; test_class2=[]; test_class3=[]; test_class4=[]; test_class5=[]; test_class6=[]; test_class7=[];
for iter =1:test_data_size
    if test_data(iter,11) == 1
        test_class1 = cat(1,test_class1,test_data(iter,2:10));
    elseif test_data(iter,11) == 2
        test_class2 = cat(1,test_class2,test_data(iter,2:10));
    elseif test_data(iter,11) == 3
        test_class3 = cat(1,test_class3,test_data(iter,2:10));
    elseif test_data(iter,11) == 4
        test_class4 = cat(1,test_class4,test_data(iter,2:10));    
    elseif test_data(iter,11) == 5
        test_class5 = cat(1,test_class5,test_data(iter,2:10));
    elseif test_data(iter,11) == 6
        test_class6 = cat(1,test_class6,test_data(iter,2:10));
    elseif test_data(iter,11) == 7
        test_class7 = cat(1,test_class7,test_data(iter,2:10));
    end
end

%For generating the summary map from all the classes for Training Data
classMap = containers.Map([1 2 3 4 5 6 7],{train_class1 train_class2 train_class3 train_class4 train_class5 train_class6 train_class7});

%Calculate the std deviation and mean of all the features for all the classes in Training Data set
std_all_attr_in_class = containers.Map('KeyType','int32','ValueType','any');
mean_all_attr_in_class = containers.Map('KeyType','int32','ValueType','any');
for iter=1:7
    if size(classMap(iter),1) ~= 0
        std_all_attr_in_class(iter) = std(classMap(iter),1,1);
        mean_all_attr_in_class(iter) = mean(classMap(iter),1);
    end
end

index_of_class = [];
disp('Summary Result for Test Data')
fprintf('\nJoint and Marginal Probability Density for each individual attributes is\n')
for test_iter=1:test_data_size
    joint_prob_row_and_class = [];
    for iter=1:7
        if size(classMap(iter),1) ~= 0 
            likelihood_prob_row_given_class = normpdf(test_data(test_iter,2:10), mean_all_attr_in_class(iter),std_all_attr_in_class(iter));
            dummy_vector=likelihood_prob_row_given_class;
            dummy_vector(isnan(dummy_vector)) = 1;
            product_likelihood = prod(dummy_vector);
            prior_prob_class = size(classMap(iter),1)/training_data_size;
            joint_prob_row_and_class = cat(2,joint_prob_row_and_class,product_likelihood*prior_prob_class);
        else
            joint_prob_row_and_class = cat(2,joint_prob_row_and_class,0);
        end
    end
    [joint_pdf_row_and_class,argmax] = max(joint_prob_row_and_class);
    index_of_class = [index_of_class;argmax];
    marginal_prob_row(test_iter) = nansum(joint_prob_row_and_class);
    disp(['Joint PDF(CLASS,ROW',num2str(test_iter),') is ',num2str(joint_pdf_row_and_class),' With Marginal PDF of ROW ',num2str(marginal_prob_row(test_iter))]);
end

accuracy_prediction = (index_of_class == test_data(:,11));
test_accuracy = sum(accuracy_prediction)*100/size(accuracy_prediction,1);
length_of_class = [size(test_class1,1),size(test_class2,1),size(test_class3,1),size(test_class4,1),size(test_class5,1),size(test_class6,1),size(test_class7,1)];
for attr=1:9
    attr_sum = 0;    
    for iter=1:7
        if size(classMap(iter),1) ~= 0
            mean_of_attrs = mean_all_attr_in_class(iter);
            std_of_attrs = std_all_attr_in_class(iter);
            P_attr_given_class = max(normpdf(test_data((1:length_of_class(iter)),attr+1), mean_of_attrs(attr),std_of_attrs(attr)));
            attr_prob(attr) = P_attr_given_class;
            prior_prob_class = size(classMap(iter),1)/training_data_size;
            attr_sum = attr_sum + nansum(P_attr_given_class *prior_prob_class);
        end
    end
    prob_attr(attr) = attr_sum;
end
fprintf('\nProbability Density for each individual attributes is\n')
attr_names = {'RI: refractive index','Na: Sodium','Mg: Magnesium','Al: Aluminum','Si: Silicon','K: Potassium','Ca: Calcium','Ba: Barium','Fe: Iron'};
for i=1:9
    name = char(attr_names{i});    
    disp(['ATTRIBUTE NAME: "',name,'" whose PROBABILITY DENSITY -> [',num2str(prob_attr(i)),']'])
end