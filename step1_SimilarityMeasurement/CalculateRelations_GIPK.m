load("CD.mat");

inter = CD;
nc = size(CD,1);
nd = size(CD,2);

[circRNA_gipk,disease_gipk]=gipk(nc,nd,inter);
save("circRNA_gipk.mat", "circRNA_gipk");
save("disease_gipk.mat", "disease_gipk");

function [result_circ,result_dis]=gipk(nc,nd,inter)

normSum=0;
    for i=1:nc  
        normSum = normSum + norm(inter(i,:))^2; 
    end
    gamal=1/(normSum/nc);
    [pkl]=zeros(nc,nc);
    for i=1:nc
        for j=1:nc
            pkl(i,j)=exp(-gamal*(norm(inter(i,:)-inter(j,:)))^2); 
        end
    end  
    
    %%disease
    normSum=0;
    for i=1:nd
        normSum = normSum + norm(inter(:,i))^2;
    end
    gamad=1/(normSum/nd);
    [pkd]=zeros(nd,nd);
    for i=1:nd
        for j=1:nd
            pkd(i,j)=exp(-gamad*(norm(inter(:,i)-inter(:,j)))^2);
        end
    end 
    result_circ=pkl;
    result_dis=pkd;
    
end