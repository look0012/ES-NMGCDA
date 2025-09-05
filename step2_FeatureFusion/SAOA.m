function [gbest_fit, gbest, Convergence_curve] = SAOA(pop_size, max_iter, lb, ub, dim, fobj)
    % 初始化种群
    pop = lb + (ub - lb) .* rand(pop_size, dim);
    fit = feval(fobj, pop);
    fit = fit(:);
    [gbest_fit, idx] = min(fit);
    gbest = pop(idx, :);
    
    % 算法参数初始化
    alpha = 0.98;
    state_prob = [0.2, 0.5, 0.3];
    Convergence_curve = zeros(1, max_iter);
    
    for iter = 1:max_iter
        % ========== 精英选择 ==========
        [~, sorted_idx] = sort(fit);
        elite_num = ceil(0.1*pop_size);
        elite_pop = pop(sorted_idx(1:elite_num), :);
        
        % ========== 改进点1：精英个体也参与状态更新 ==========
        all_indices = 1:pop_size;  % 所有个体都参与更新
        
        % ========== 改进点2：增强的反向学习策略 ==========
        % 生成动态反向种群（不只是简单的镜像）
        opposite_pop = elite_pop + 0.5*(ub-lb).*randn(size(elite_pop));
        opposite_pop = max(min(opposite_pop, ub), lb);  % 边界处理
        
        % 计算适应度
        opposite_fit = feval(fobj, opposite_pop);
        opposite_fit = opposite_fit(:);
        
        % 合并种群
        temp_pop1 = [pop; opposite_pop];
        temp_fit1 = [fit; opposite_fit];
        [temp_fit1, idx] = sort(temp_fit1);
        temp_pop = temp_pop1(idx(1:pop_size), :);
        temp_fit = temp_fit1(idx(1:pop_size), :);
        
        % ========== 种群多样性计算 ==========
        diversity = mean(std(temp_pop)) / norm(ub-lb);
        
        % ========== 动态状态概率调整 ==========
        state_prob(1) = max(0.1, alpha*(0.4*(1-iter/max_iter) + 0.2*diversity));
        state_prob(2) = max(0.1, 0.3 - alpha*0.15*(iter/max_iter));
        state_prob(3) = max(0.1, alpha*0.5*(iter/max_iter) + 0.3*(1-diversity));
        state_prob = state_prob / sum(state_prob);
        
        % ========== 改进点3：精英个体也参与状态更新 ==========
        for i = all_indices
            state = randsample(3, 1, true, state_prob);
            
            switch state
                case 1 % 固态：精英引导开发
                    % 改进点4：引入DE-like的精英引导
                    if i <= elite_num
                        % 精英个体使用更强的局部搜索
                        sigma = 0.05*(ub-lb)*exp(-2*alpha*iter/max_iter);
                        new_pos = gbest + sigma.*randn(1,dim);
                    else
                        sigma = 0.1*(ub-lb)*exp(-alpha*iter/max_iter);
                        new_pos = gbest + sigma.*randn(1,dim);
                    end
                    
                % 修改液态状态部分
                case 2 % 液态：自适应差分进化
                    F = 0.5 + 0.5*rand()*cos(pi*iter/max_iter);  % 增加随机性
                    CR = 0.3 + 0.5*rand();  % 更自适应的交叉率
                    idxs = randperm(pop_size, 3);
                    mutant = temp_pop(idxs(1),:) + F*(temp_pop(idxs(2),:) - temp_pop(idxs(3),:));
                    mask = rand(1,dim) < CR;
                    new_pos = temp_pop(i,:).*~mask + mutant.*mask;
                    
                case 3 % 气态：强化Levy飞行
                    beta = 1.8 - 0.6*(iter/max_iter); 
                    step_scale = 0.2*exp(-iter/max_iter);
                    step = levy_step(beta, dim);
                    new_pos = temp_pop(i,:) + step_scale*step.*(rand(1,dim)-0.5);
            end
            
            % 边界处理
            new_pos = max(min(new_pos, ub), lb);
            new_fit = feval(fobj, new_pos);
            
            % 适应度评估与更新
            if new_fit < temp_fit(i)
                temp_pop(i,:) = new_pos;
                temp_fit(i) = new_fit;
                % 更新全局最优解
                if new_fit < gbest_fit
                    gbest = new_pos;
                    gbest_fit = new_fit;
                end
            end
        end
        
        % 更新种群
        pop = temp_pop;
        fit = temp_fit;
        
        % ========== 多样性维持策略 ==========
        if mod(iter,30) == 0 || diversity < 0.03
            reset_num = ceil(0.15*pop_size);
            reset_idx = randperm(pop_size, reset_num);
            pop(reset_idx,:) = lb + (ub-lb).*rand(reset_num,dim);
            fit(reset_idx) = feval(fobj, pop(reset_idx,:));
        end
        
        alpha = alpha * 0.98;
        Convergence_curve(iter) = gbest_fit;
    end
end

% Levy飞行步长生成函数
function step = levy_step(beta, dim)
    % 计算Levy分布参数
    num = gamma(1+beta) * sin(pi*beta/2);
    den = gamma((1+beta)/2) * beta * 2^((beta-1)/2);
    sigma = (num/den)^(1/beta);
    % 生成Levy随机步长
    u = randn(1,dim) * sigma;
    v = randn(1,dim);
    step = u ./ abs(v).^(1/beta);
end