
% load('CD.mat');  % RNA序列特征
% load('CFS.mat');  % RNA序列特征
% load('MeSHSemanticSimilarity.mat');  % 疾病语义特征

A = CD;
P_feat = circRNA_CFS;
D_feat = MeSHSemanticSimilarity;

[U, V] = MDMF(A, P_feat, D_feat, 0.1, 100, 100);
circRNA_mdmf = U * U';
disease_mdmf = V * V';

save("circRNA_mdmf.mat", "circRNA_mdmf");
save("disease_mdmf.mat", "disease_mdmf");

function [U, V, W_p, W_d] = MDMF(A, protein_feat, disease_feat, lambda, max_iter, k)

    if any(isnan(A(:))) || any(isnan(protein_feat(:))) || any(isnan(disease_feat(:)))
        error('输入矩阵包含NaN值，请检查数据！');
    end
    

    protein_feat = zscore(protein_feat); 
    disease_feat = zscore(disease_feat);  
    
    [m, n] = size(A);
    p = size(protein_feat, 2);
    q = size(disease_feat, 2);
    
    U = 0.01 * randn(m, k);      
    V = 0.01 * randn(n, k);      
    W_p = 0.01 * randn(p, k);   
    W_d = 0.01 * randn(q, k);   
    
    lr = 0.001;                 
    epsilon = 1e-8;            
    
    for iter = 1:max_iter
      
        protein_proj = protein_feat * W_p;
        disease_proj = disease_feat * W_d;
        
      
        UV = U * V';
        UV = max(min(UV, 1e4), -1e4); 
        reconstruction_loss = UV - A;
        
      
        grad_U = reconstruction_loss * V + lambda * (U - protein_proj);
        grad_V = reconstruction_loss' * U + lambda * (V - disease_proj);
        grad_Wp = protein_feat' * (U - protein_proj) * lambda + 1e-4 * W_p;  % L2正则化
        grad_Wd = disease_feat' * (V - disease_proj) * lambda + 1e-4 * W_d;
        

        grad_U = sign(grad_U) .* min(abs(grad_U), 1e3);
        grad_V = sign(grad_V) .* min(abs(grad_V), 1e3);
        grad_Wp = sign(grad_Wp) .* min(abs(grad_Wp), 1e3);
        grad_Wd = sign(grad_Wd) .* min(abs(grad_Wd), 1e3);
        
   
        if iter == 1
            momentum_U = 0;
            momentum_V = 0;
            momentum_Wp = 0;
            momentum_Wd = 0;
        else
            momentum_U = 0.9 * momentum_U + lr * grad_U;
            momentum_V = 0.9 * momentum_V + lr * grad_V;
            momentum_Wp = 0.9 * momentum_Wp + lr * grad_Wp;
            momentum_Wd = 0.9 * momentum_Wd + lr * grad_Wd;
        end
        
        U = U - momentum_U;
        V = V - momentum_V;
        W_p = W_p - momentum_Wp;
        W_d = W_d - momentum_Wd;
        
      
        U = max(min(U, 1e3), -1e3);
        V = max(min(V, 1e3), -1e3);
        W_p = max(min(W_p, 1e3), -1e3);
        W_d = max(min(W_d, 1e3), -1e3);
        
       
        if any(isnan(U(:))) || any(isnan(V(:)))
            error(['迭代 ', num2str(iter), ': 检测到NaN，终止训练！']);
        end
    end
end