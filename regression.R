setwd("~/code/SamplingDynamics/")

library(lme4)

data = as.data.frame(read.csv('reg_data.csv'))
data$domain = factor(data$domain)
data$gamble_lab = factor(data$gamble_lab)
data$switched = factor(data$switched)
data$group = factor(data$group)
data$sample_mean = as.numeric(as.character(data$sample_mean))
data$deviation = as.numeric(as.character(data$deviation))

dims = list(unique(data$partid), c("partid", "m1_aic"))
df = data.frame(matrix(vector(), 12, 2, dimnames=dims), stringsAsFactors=F)

domainloss = vector()

for (sid in unique(data$partid)) {
  sdata = data[data$partid==sid,]  
  
  m1 = glmer(switched ~ 1 + domain + (1|gamble_lab), data=sdata, family=binomial)
  print(summary(m1))
  
  domainloss = c(domainloss, fixef(m1)['domainloss'])
  
  print(fixef(m1)['domainloss'])
  
}

sdata = data[data$partid==3,]

m1 = glmer(switched ~ 1 + domain + (1|gamble_lab), data=sdata, family=binomial)
m2 = glmer(switched ~ 1 + domain + streak_length + (1|gamble_lab), data=sdata, family=binomial)
m3 = glmer(switched ~ 1 + domain + sample_mean + deviation + (1|gamble_lab), data=sdata, family=binomial)




sum# for each participant, run a series of logistic regressions

