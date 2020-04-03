
#Name:Shreya Banerjee
#SRN:PES1201700198

#Packages required
require(data.table)
require(plotrix)



path='/Users/shreyabanerjee/Assignment1'





data1=fread("general_data.csv")
data2=fread("employee_survey_data.csv")
data3=fread("manager_survey_data.csv")
data1=na.omit(data1)      #omitting Na values
data2=na.omit(data2)
data3=na.omit(data3)
data4=merge(data1,data2)   #merging two files
data=merge(data4,data3)    #merging the third file



#<h2>Question 1)a.</h2>
 
hist(data$Age,xlim=c(20,60),ylim=c(0,2000),xlab="Age",breaks=5,las=1,col="blue")
#<b>The graph is skewed to the right(positively skewed).</b>
  
  
  

avg_incomes=c(0,0,0,0,0)
#calculating average incomes for all the age intervals beginning from 20-29
for(i in 2:5){
  avg_incomes[i]=mean(data$MonthlyIncome[data$Age>=i*10 & data$Age<i*10+10])
}
print(avg_incomes)
print(max(avg_incomes))

#<b>40-49 age group has the highest average monthly income.</b>
  
#<h2>Question 1)b.</h2>
#sub is a table containing records(income,job and working years) of employees within the range 40-49
sub =subset(data, data$Age >= 40 & data$Age<50,select=c(MonthlyIncome,JobRole,TotalWorkingYears))
sub_sorted=sub[order(sub$MonthlyIncome)]    #sorting the incomes in increasing order
incomes=unique(sub_sorted$MonthlyIncome)    #obtaining unique values of incomes
fifth_highest=incomes[length(incomes)-4]    #all the salaries greater than the fifth highest                                                 salaries required
sub =subset(sub_sorted, sub_sorted$MonthlyIncome >=fifth_highest)  #sub contains the details of                                                            employees with 5 highest salaries
print(sub)




#<h2>Question 1)c</h2>

#subs is a table containing records(income and job role) of employees within the range 40-49
subs =subset(data, data$Age >= 40 & data$Age < 49,select=c(MonthlyIncome,JobRole))
a=aggregate(subs$MonthlyIncome, by=list(subs$JobRole), FUN=mean)
print(a)
#a contains the average income of each job role
avg_incomes=a$x
job_roles=a$Group.1
print(job_roles)
print(avg_incomes)
par(mar=c(11,5,1,5))
barplot(avg_incomes,names.arg=job_roles,density=100,las=2,ylab="Average Monthly Income",col=c("red"),ann=FALSE)
mtext(side = 1, text = "Job Roles", line = 10)
mtext(side = 2, text = "Average Monthly Income", line = 4)






#<h2>Question 2)</h2>
counts <- table(data$EnvironmentSatisfaction, data$MaritalStatus)
print(counts)
barplot(counts,ylim=c(0,2500),ylab="Number of women",xlab="Marital Status",col=c("red","yellow","green","blue"),legend.text = TRUE)


#<h2>Question 3)</h2>


tab1=data.frame(table(data$Department))   #grouping the data department-wise
dept=tab1$Var1                            #departments
in_each_dept=tab1$Freq             #number of employees in each department
print(tab1)
a=subset(data,data$Attrition=="Yes")    #contains tuples for which attrition is yes
tab=data.frame(table(a$Department))    #number of employees with attrition in each department
print(tab)
att_yes=tab$Freq                       #number of employees with attrition in each department 

perc=c(0,0,0)
for (i in 1:length(in_each_dept)){
  perc[i] <- att_yes[i]*100/in_each_dept[i]     #percentage of employees with attrition in each                                                  department 
}
A=min(perc)
ind=which(perc==A)                            #finding index of department with minimum attrition
print(dept[ind])   


#<b>Sales department has least attrition in terms of percentage of employees in that department.</b>
  
#<h2>Question 4)a</h2>

par(mar=c(8,5,1,1))
boxplot(data$MonthlyIncome~data$EducationField,las=2,ann=FALSE)
mtext(side = 1, text = "Educational Field", line = 6)
mtext(side = 2, text = "Monthly Income", line = 4)

tapply(data$MonthlyIncome, data$EducationField, summary)


#<h2>Question 4)b</h2>

women_avg_income=mean(data$MonthlyIncome[data$Gender=="Female"])
men_avg_income=mean(data$MonthlyIncome[data$Gender=="Male"])
men_avg_income-women_avg_income

#<b>Gender pay gap is Rs.589.5354.</b>
  
  
  
#<h2>Question 4)c</h2>

aggregate(data$MonthlyIncome, by = list(data$EducationField), FUN = quantile, probs  = 0.95)


#<b>183000 is the 95th percentile monthly income for the Medical Education Field.</b>
  
  
  
  
  
#<h2>Question 5)</h2>
#Skewness is the measure of symmetry.
#It is the degree of distortion from the normal distribution. A symmetrical distribution will have a skewness of 0.
#Positive Skewness- when the tail on the right side of the distribution is longer or fatter. 
#Negative Skewness- when the tail of the left side of the distribution is longer or fatter than the tail on the right side.

#Kurtosis is the measure of peakedness.
#High kurtosis in a data set is an indicator that data has heavy tails or outliers. 
#Low kurtosis in a data set is an indicator that data has light tails or lack of outliers. 

