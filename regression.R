setwd("~/code/SamplingDynamics/")

library(lme4)

data = as.data.frame(read.csv('reg_data.csv'))

m = glmer(switch ~ gamble_lab + (1|partid), data=data, family=binomial)
