
load CD.mat

dim = 30;
[circRNA_ges, disease_ges] = graph_embedding_similarity(CD, dim);
save("circRNA_ges.mat", "circRNA_ges");
save("disease_ges.mat", "disease_ges");


function [rna_sim, disease_sim] = graph_embedding_similarity(CD, dim)

    if nargin < 2
        dim = 20;
    end
    

    [m, n] = size(CD);
    

    A = sparse([zeros(m,m), CD; CD', zeros(n,n)]); 
    

    D = sum(A, 2);
    D(D == 0) = eps; 
    D_inv_sqrt = diag(1 ./ sqrt(D));
    L_norm = speye(size(A)) - D_inv_sqrt * A * D_inv_sqrt; 
    

    sigma = 1e-6; 
    max_dim = min(size(L_norm,1)-1, dim+1); 
    
    opts.tol = 1e-8;
    opts.maxit = 500;
    [V, ~] = eigs(L_norm, max_dim, sigma, opts); 
    
    actual_dim = size(V,2) - 1; 
    if actual_dim < 1
        error('无法提取有效嵌入，请降低dim值或检查输入矩阵');
    end
    embedding = V(:, 2:min(end,dim+1)); 
    

    rna_embed = embedding(1:m, :);
    disease_embed = embedding(m+1:end, :);
    
    rna_sim = (1 - pdist2(rna_embed, rna_embed, 'cosine')) .* (1 - eye(m));
    disease_sim = (1 - pdist2(disease_embed, disease_embed, 'cosine')) .* (1 - eye(n));
    
    rna_sim = max((rna_sim + rna_sim')/2, 0);
    disease_sim = max((disease_sim + disease_sim')/2, 0);
end