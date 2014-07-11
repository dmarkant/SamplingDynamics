library(ggplot2)

ggplot(rare_data, aes(abs_dev, color=switch_or_stop)) + 
  geom_density(alpha = .2)

ggplot(rare_data, aes(sample_out, color=switch_or_stop)) + 
  geom_density(alpha = .2)