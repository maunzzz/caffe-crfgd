%   Log parser specifically written for crfgd training
%
%   
%   Måns Larsson 20180619


%train - for parsing logs during training of netwowk with crf module as final layer
type_of_log = 'train'; %'pretrain', 'train'
log_file = '../training/crf_ed/log.log';
%log_file = '../training/filterbank/log_pretrain.log';
%log_file = '../training/filterbank/log_train.log';


%% PARSING
fid = fopen(log_file);

train_iteration = [];
train_loss = [];

val_iteration = [];
val_iou = [];
val_edge_loss = [];

tline = fgetl(fid);
curr_iter = 0;
while ischar(tline)
    disp(tline)
    
    if ~isempty(strfind(tline,'solver.cpp:239'))
        it_pos = strfind(tline,'Iteration');
        pos = strfind(tline,'(');
        loss_pos = strfind(tline,'loss');
        
        curr_iter = str2double(tline(it_pos+10:pos(1)-2));
        train_iteration = [train_iteration curr_iter];
        train_loss = [train_loss str2double(tline(loss_pos+7:end))];
    end
    
    if ~isempty(strfind(tline,'iou_accuracy_layer.cpp:152'))
        a = strfind(tline,'Accuracy');
        
        val_iteration = [val_iteration curr_iter];
        val_iou = [val_iou str2double(tline(a+9:end))];
    end
    
    if strcmpi(type_of_log,'pretrain')
        if ~isempty(strfind(tline,'Test net output #0'))
            a = strfind(tline,'=');
            b = strfind(tline,'(');
            val_edge_loss = [val_edge_loss str2double(tline(a(1)+2:b(1)-2))];
        end
    end
    
    
    tline = fgetl(fid);
end
fclose(fid);

%% PLOTTING
slashinds = strfind(log_file,'/');
datasetname = log_file(slashinds(end-1)+1:slashinds(end)-1);
log_name = log_file(slashinds(end)+1:end-4);
figname = sprintf('%s - %s',datasetname,log_name);
figure('name',figname);

subplot(1,3,1)
plot(train_iteration,train_loss)
title('training loss')
xlabel('it')

if strcmpi(type_of_log,'train')
    subplot(1,3,2)
    plot(val_iteration(1:2:end),val_iou(1:2:end))
    title({'val iou post', num2str(max(val_iou(1:2:end)))})
    xlabel('it')
    
    subplot(1,3,3)
    plot(val_iteration(2:2:end),val_iou(2:2:end))
    title({'val iou pre', num2str(max(val_iou(2:2:end)))})
    xlabel('it')
    
    post_val = val_iou(1:2:end);
    it = val_iteration(1:2:end);
    [m,i] = max(post_val);
    fprintf('best iteration %d \n',it(i));
end

if strcmpi(type_of_log,'pretrain')
    subplot(1,3,2)
    plot(val_iteration,val_iou)
    title({'val iou seg', num2str(max(val_iou))})
    xlabel('it')
    
    subplot(1,3,3)
    plot(val_iteration,val_edge_loss)
    title({'val edge loss', num2str(min(val_edge_loss))})
    xlabel('it')
end
