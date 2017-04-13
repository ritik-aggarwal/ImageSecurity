function E = Arnold_map_decrypt(C)
M = 512;
N = 512;
E = zeros(M,N,3);
for itr = 1:63
    for k = 1:3
        for i = 1:M
            for j = 1:N
                newi = mod((i - j),M)+1;
                newj = mod((2*j - i),N)+1;
                E(newi,newj,k) = C(i,j,k);
            end
        end
    end
    C = E;
end
end