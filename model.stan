data {
  int<lower=1> numPred;
  int<lower=0> N;
  int<lower=1> numPeople;
  int<lower=1,upper=3> y[N];
  int<lower=1,upper=numPeople> person[N];
  int<lower=0,upper=1> gender[numPeople];
  int<lower=0,upper=1> age[numPeople];
  row_vector[numPred] x[N];
}
parameters {
  row_vector[numPred] mu;
  row_vector[numPred] male_mu;
  row_vector[numPred] older_mu;
  row_vector<lower=0>[numPred] sigma;
  row_vector[numPred] person_predictor_raw[numPeople];
  ordered[2] c;
}
transformed parameters {
  row_vector[numPred] person_predictor[numPeople];
  vector[N] x_beta_ll;
  for (l in 1:numPeople) {
    person_predictor[l] = mu + male_mu * (gender[l] * 2 - 1) + older_mu  * (age[l] * 2 - 1) + person_predictor_raw[l] .* sigma;
  }

  for (n in 1:N)
    x_beta_ll[n] = x[n] * person_predictor[person[n]]';
}
model {
  mu ~ normal(0, 10);
  male_mu ~ normal(0, 10);
  older_mu ~ normal(0, 10);
  for (l in 1:numPeople) {
    person_predictor_raw[l] ~ std_normal();
  }
  y ~ ordered_logistic(x_beta_ll, c);
}
