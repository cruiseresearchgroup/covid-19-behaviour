library(tidyverse)
library("rstan") 
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

segments <- read_csv("out.csv")
users <- read_csv("users.csv")
users$code <- factor(users$code)
users$codeno <- as.numeric(users$code)
users <- arrange(users, codeno)
full <- inner_join(segments, users, by="code")
full$code <- factor(full$code)
full$codeno <- as.numeric(full$code)
full = filter(full, happiness != -1)
full = filter(full, alertness != -1)
full$happiness[full$happiness == 2] <- 1
full$happiness[full$happiness == 3] <- 2
full$happiness[full$happiness == 4] <- 3
full$happiness[full$happiness == 5] <- 3
full$alertness[full$alertness == 2] <- 1
full$alertness[full$alertness == 3] <- 2
full$alertness[full$alertness == 4] <- 3
full$alertness[full$alertness == 5] <- 3
print(full$happiness)

people <- unique(segments$code)

model <- list(numPred=16,
              N = nrow(full),
              numPeople = nrow(users),
              y = full$alertness,
              person = full$codeno,
              gender = users$gender,
              age = (users$age <= 24),
              x = select(full, `0`, `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`, `10`, `11`, `12`, `13`, `14`, `15`)
              )

fit <- stan(file = 'model.stan', data = model, iter=1000, chains=2)
print(fit)
saveRDS(fit, "fitalert.rds")
#plot(fit)
