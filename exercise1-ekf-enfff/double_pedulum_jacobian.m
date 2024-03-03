clc
clear
close all

syms x [4 1]

% x1 = th1 angle 1
% x2 = th2 angle 2
% x3 = angular velocity 1
% x4 = angular velocity 2

th1 = x1;
th2 = x2;
thdot1 = x3;
thdot2 = x4;

g = -9.81;
l1 = 1;
l2 = 1;
m1 = 1;
m2 = 1;

deltaTime = 0.01;

beta = [
  x3
  x4
  -g*(2*m1+m2)+sin(th1)-m2*g*sin(th1-2*th2)-2*sin(th1-th2)*m2*(thdot2^2*l2+thdot1^2*l2*cos(th1-th2))/(l1*(2*m1+m2-m2*cos(2*th1-2*th2)))
  2*sin(th1-th2)*(thdot1^2*l1*(m1+m2)+g*(m1+m2)*cos(th1)+thdot2^2*l2*m2*cos(th1-th2))/(l1*(2*m1+m2-m2*cos(2*th1-2*th2)))
];

jac = jacobian(x + deltaTime*beta, x)
latex(jac)

 
% jac =
% 
% [                                                                                                                                                                                                                                                                                                 1,                                                                                                                                                                                                                                                              0,                                                          1/100,                                                              0]
% [                                                                                                                                                                                                                                                                                                 0,                                                                                                                                                                                                                                                              1,                                                              0,                                                          1/100]
% [                                   (981*cos(x1 - 2*x2))/10000 + cos(x1)/100 + (cos(x1 - x2)*(cos(x1 - x2)*x3^2 + x4^2))/(50*(cos(2*x1 - 2*x2) - 3)) - (x3^2*sin(x1 - x2)^2)/(50*(cos(2*x1 - 2*x2) - 3)) + (sin(x1 - x2)*sin(2*x1 - 2*x2)*(cos(x1 - x2)*x3^2 + x4^2))/(25*(cos(2*x1 - 2*x2) - 3)^2),               (x3^2*sin(x1 - x2)^2)/(50*(cos(2*x1 - 2*x2) - 3)) - (cos(x1 - x2)*(cos(x1 - x2)*x3^2 + x4^2))/(50*(cos(2*x1 - 2*x2) - 3)) - (981*cos(x1 - 2*x2))/5000 - (sin(x1 - x2)*sin(2*x1 - 2*x2)*(cos(x1 - x2)*x3^2 + x4^2))/(25*(cos(2*x1 - 2*x2) - 3)^2), (x3*cos(x1 - x2)*sin(x1 - x2))/(25*(cos(2*x1 - 2*x2) - 3)) + 1,                  (x4*sin(x1 - x2))/(25*(cos(2*x1 - 2*x2) - 3))]
% [- (sin(x1 - x2)*(- sin(x1 - x2)*x4^2 + (981*sin(x1))/50))/(50*(cos(2*x1 - 2*x2) - 3)) - (cos(x1 - x2)*(2*x3^2 + cos(x1 - x2)*x4^2 - (981*cos(x1))/50))/(50*(cos(2*x1 - 2*x2) - 3)) - (sin(x1 - x2)*sin(2*x1 - 2*x2)*(2*x3^2 + cos(x1 - x2)*x4^2 - (981*cos(x1))/50))/(25*(cos(2*x1 - 2*x2) - 3)^2), (cos(x1 - x2)*(2*x3^2 + cos(x1 - x2)*x4^2 - (981*cos(x1))/50))/(50*(cos(2*x1 - 2*x2) - 3)) - (x4^2*sin(x1 - x2)^2)/(50*(cos(2*x1 - 2*x2) - 3)) + (sin(x1 - x2)*sin(2*x1 - 2*x2)*(2*x3^2 + cos(x1 - x2)*x4^2 - (981*cos(x1))/50))/(25*(cos(2*x1 - 2*x2) - 3)^2),               -(2*x3*sin(x1 - x2))/(25*(cos(2*x1 - 2*x2) - 3)), 1 - (x4*cos(x1 - x2)*sin(x1 - x2))/(25*(cos(2*x1 - 2*x2) - 3))]
% 
% 
% ans =
% 
%     '\left(\begin{array}{cccc} 1 & 0 & \frac{1}{100} & 0\\ 0 & 1 & 0 & \frac{1}{100}\\ \frac{981\,\cos\left(x_{1}-2\,x_{2}\right)}{10000}+\frac{\cos\left(x_{1}\right)}{100}+\frac{\cos\left(x_{1}-x_{2}\right)\,\left(\cos\left(x_{1}-x_{2}\right)\,{x_{3}}^2+{x_{4}}^2\right)}{50\,\left(\cos\left(2\,x_{1}-2\,x_{2}\right)-3\right)}-\frac{{x_{3}}^2\,{\sin\left(x_{1}-x_{2}\right)}^2}{50\,\left(\cos\left(2\,x_{1}-2\,x_{2}\right)-3\right)}+\frac{\sin\left(x_{1}-x_{2}\right)\,\sin\left(2\,x_{1}-2\,x_{2}\right)\,\left(\cos\left(x_{1}-x_{2}\right)\,{x_{3}}^2+{x_{4}}^2\right)}{25\,{\left(\cos\left(2\,x_{1}-2\,x_{2}\right)-3\right)}^2} & \frac{{x_{3}}^2\,{\sin\left(x_{1}-x_{2}\right)}^2}{50\,\left(\cos\left(2\,x_{1}-2\,x_{2}\right)-3\right)}-\frac{\cos\left(x_{1}-x_{2}\right)\,\left(\cos\left(x_{1}-x_{2}\right)\,{x_{3}}^2+{x_{4}}^2\right)}{50\,\left(\cos\left(2\,x_{1}-2\,x_{2}\right)-3\right)}-\frac{981\,\cos\left(x_{1}-2\,x_{2}\right)}{5000}-\frac{\sin\left(x_{1}-x_{2}\right)\,\sin\left(2\,x_{1}-2\,x_{2}\right)\,\left(\cos\left(x_{1}-x_{2}\right)\,{x_{3}}^2+{x_{4}}^2\right)}{25\,{\left(\cos\left(2\,x_{1}-2\,x_{2}\right)-3\right)}^2} & \frac{x_{3}\,\cos\left(x_{1}-x_{2}\right)\,\sin\left(x_{1}-x_{2}\right)}{25\,\left(\cos\left(2\,x_{1}-2\,x_{2}\right)-3\right)}+1 & \frac{x_{4}\,\sin\left(x_{1}-x_{2}\right)}{25\,\left(\cos\left(2\,x_{1}-2\,x_{2}\right)-3\right)}\\ -\frac{\sin\left(x_{1}-x_{2}\right)\,\left(\frac{981\,\sin\left(x_{1}\right)}{50}-{x_{4}}^2\,\sin\left(x_{1}-x_{2}\right)\right)}{50\,\left(\cos\left(2\,x_{1}-2\,x_{2}\right)-3\right)}-\frac{\cos\left(x_{1}-x_{2}\right)\,\left(2\,{x_{3}}^2+\cos\left(x_{1}-x_{2}\right)\,{x_{4}}^2-\frac{981\,\cos\left(x_{1}\right)}{50}\right)}{50\,\left(\cos\left(2\,x_{1}-2\,x_{2}\right)-3\right)}-\frac{\sin\left(x_{1}-x_{2}\right)\,\sin\left(2\,x_{1}-2\,x_{2}\right)\,\left(2\,{x_{3}}^2+\cos\left(x_{1}-x_{2}\right)\,{x_{4}}^2-\frac{981\,\cos\left(x_{1}\right)}{50}\right)}{25\,{\left(\cos\left(2\,x_{1}-2\,x_{2}\right)-3\right)}^2} & \frac{\cos\left(x_{1}-x_{2}\right)\,\left(2\,{x_{3}}^2+\cos\left(x_{1}-x_{2}\right)\,{x_{4}}^2-\frac{981\,\cos\left(x_{1}\right)}{50}\right)}{50\,\left(\cos\left(2\,x_{1}-2\,x_{2}\right)-3\right)}-\frac{{x_{4}}^2\,{\sin\left(x_{1}-x_{2}\right)}^2}{50\,\left(\cos\left(2\,x_{1}-2\,x_{2}\right)-3\right)}+\frac{\sin\left(x_{1}-x_{2}\right)\,\sin\left(2\,x_{1}-2\,x_{2}\right)\,\left(2\,{x_{3}}^2+\cos\left(x_{1}-x_{2}\right)\,{x_{4}}^2-\frac{981\,\cos\left(x_{1}\right)}{50}\right)}{25\,{\left(\cos\left(2\,x_{1}-2\,x_{2}\right)-3\right)}^2} & -\frac{2\,x_{3}\,\sin\left(x_{1}-x_{2}\right)}{25\,\left(\cos\left(2\,x_{1}-2\,x_{2}\right)-3\right)} & 1-\frac{x_{4}\,\cos\left(x_{1}-x_{2}\right)\,\sin\left(x_{1}-x_{2}\right)}{25\,\left(\cos\left(2\,x_{1}-2\,x_{2}\right)-3\right)} \end{array}\right)'
