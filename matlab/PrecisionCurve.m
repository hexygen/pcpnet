function [ e, p ] = PrecisionCurve( err )
% PRECISIONCURVE Creates a precision curve where each point signifies the
% percent of data points with an error less than X.

k = 1000;
n = length(err);

m = max(err);

e = 0;
p = 0;

for i = 1:k
    % Current error level:
    e(i+1) = m/k * i;
    % Matching percent:
    p(i+1) = sum(err < e(i+1)) / n;    
end;


end

