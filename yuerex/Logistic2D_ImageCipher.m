function varargout = Logistic2D_ImageCipher(P,para,K)

%% 1. Initialization
% 1.1. Genereate Random Key if K is not given
if ~exist('K','var') && strcmp(para,'encryption') 
    K = round(rand(1,256));
    varOutN = 2;
elseif ~exist('K','var') && strcmp(para,'decryption') 
    error('Cannot Complete Decryption without Encryption Key')
else
    varOutN = 1;
end
% 1.2. Translate K to map formats
transFrac = @(K,st,ed) sum(K(st:ed).*2.^(-(1:(ed-st+1))));
x0 = transFrac(K,1,52);
y0 = transFrac(K,53,104);
r = transFrac(K,105,156)*.08+1.11;
T = transFrac(K,157,208);
turb = blkproc(K(209:256),[1,8],@(x) bi2de(x));

MN = numel(P);
% 1.3 2D Logistic map
Logistic2D = @(x,y,r) [r*(3*y+1)*x*(1-x), r*(3*(r*(3*y+1)*x*(1-x))+1)*y*(1-y)];
format long eng
%% 2. Estimate cipher rounds
if max(P(:))>1
    F = 256;
    S = 4;
else
    F = 2;
    S = 32;
end
P = double(P);
iter = ceil(log2(numel(P))/log2(S));

%% 3. Image Cipher
C = double(P);
switch para
    case 'encryption'
        for i = 1:iter
            tx0 = mod(log(turb(mod(i-1,6)+1)+i)*x0+T,1);
            ty0 = mod(log(turb(mod(i-1,6)+1)+i)*y0+T,1);
            xy = zeros(MN,2);
            for n = 1:MN
                if n == 1
                    xy(n,:) = abs(Logistic2D(tx0,ty0,r)); 
                else
                    xy(n,:) = abs(Logistic2D(xy(n-1,1),xy(n-1,2),r));
                end
            end
            R = cat(3,reshape(xy(:,1),size(P,1),size(P,2)),reshape(xy(:,2),size(P,1),size(P,2)));
            C = LogisticSubstitution(C,R,'encryption');   
            C = LogisticPermutation(C,R,'encryption');
            C = LogisticMDS(C,'encryption');
        end
    case 'decryption'
        for i = iter:-1:1
            tx0 = mod(log(turb(mod(i-1,6)+1)+i)*x0+T,1);
            ty0 = mod(log(turb(mod(i-1,6)+1)+i)*y0+T,1);
            xy = zeros(MN,2);
            for n = 1:MN
                if n == 1
                    xy(n,:) = abs(Logistic2D(tx0,ty0,r)); 
                else
                    xy(n,:) = abs(Logistic2D(xy(n-1,1),xy(n-1,2),r));
                end
            end
            R = cat(3,reshape(xy(:,1),size(P,1),size(P,2)),reshape(xy(:,2),size(P,1),size(P,2)));
            C = LogisticMDS(C,'decryption');
            C = LogisticPermutation(C,R,'decryption'); 
            C = LogisticSubstitution(C,R,'decryption');                          
        end
end

%% 4. Output
switch F
    case 2
        C = logical(C);
    case 256
        C = uint8(C);
end

switch varOutN
    case 1
        varargout{1} = C;
    case 2
        varargout{1} = C;
        varargout{2} = K;
end