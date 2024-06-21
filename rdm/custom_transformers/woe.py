import logging
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from pandas.api.types import is_datetime64_any_dtype
from sklearn.utils.random import check_random_state


class WOEEncoder(BaseEstimator, TransformerMixin):
    """
        Weight of Evidence (WoE) Encoder for binary classification problems.

        This encoder transforms categorical variables into numerical values using the Weight of Evidence method,
        which measures the predictive power of a categorical variable in relation to the binary target variable.

        Parameters
        ----------
        bins : int, default=10
            The number of bins to use for binning numerical variables. Only applicable to numerical features.

        min_samples : float, default=0.05
            The minimum sample size for each bin, expressed as a fraction of the total sample size. Helps to ensure
            that bins have enough samples and are not too granular.

        retain_only_predictive_features : bool, default=False
            If set to True, only features with Information Value (IV) within the range of 0.02 to 0.5 are retained,
            dropping features that are either not useful for prediction or have suspiciously high predictive power.

        regularization : float, default=1.0
            Regularization value added to the numerator and denominator of the WoE formula to prevent division by zero.

        randomized : bool, default=False
            If set to True, adds random Gaussian noise to the encoded features during the transform phase to reduce
            overfitting. The noise is not added during the fit phase.

        sigma : float, default=0.05
            The standard deviation of the Gaussian noise added to the features when randomized is set to True.

        drop_invariant : bool, default=False
            If set to True, drops features that do not change in value (invariant features) after encoding.

        random_state : int, RandomState instance, default=None
            The seed of the pseudo random number generator to use when shuffling the data and adding noise.
            Pass an int for reproducible output across multiple function calls.

        Methods
        -------
        fit(X, y):
            Fits the encoder to the data, computing the WoE and IV values for each feature.

        transform(X):
            Transforms the data using the fitted WoE values. If randomized is True, adds Gaussian noise to the features.

        fit_transform(X, y):
            Fits the encoder and transforms the dataset in one step.

        retain_predictive_features():
            Retains only features with IV values in the range of 0.02 to 0.5.

        drop_invariant_features():
            Drops features that are invariant after encoding.

        Examples
        --------
            encoder = WOEEncoder(bins=10, regularization=1.0, randomized=True, sigma=0.05)
            X_encoded = encoder.fit_transform(X, y)
        """

    def __init__(self, bins=10, min_samples=0.05, retain_only_predictive_features=True,
                 regularization=1.0, randomized=False, sigma=0.05, drop_invariant=False, random_state=42):
        self.bins = bins
        self.min_samples = min_samples
        self.retain_only_predictive_features = retain_only_predictive_features
        self.woe = pd.DataFrame()
        self.iv = pd.DataFrame()
        self.regularization = regularization
        self.randomized = randomized
        self.sigma = sigma
        self.drop_invariant = drop_invariant
        self.random_state = random_state

    def _bin_data(self, series, y) -> pd.DataFrame:
        """Bins the data and calculates aggregated counts and sums."""
        if series.dtype.kind in 'bifc' and len(series.unique()) > self.bins:
            binned_series = pd.qcut(series, self.bins, duplicates='drop')
        else:
            binned_series = series
        grouped = pd.DataFrame({'x': binned_series, 'y': y}).groupby('x', as_index=False).agg({'y': ['count', 'sum']})
        grouped.columns = ['Cutoff', 'N', 'Events']
        return grouped

    def fit(self, X, y):
        """
        Fits the encoder to the data.

        Parameters
        ----------
        X : Pandas DataFrame, shape [n_samples, n_features]
            The data to encode.
        y : Pandas DataFrame or Pandas Series, shape [n_samples]
            The binary target variable.

        Returns
        -------
        self : encoder
            Returns the instance itself.
        """
        logging.info("Starting fit method.")
        # Check if y is binary
        y = pd.Series(y)
        if len(y.unique()) != 2:
            raise ValueError("Target variable (y) must be binary.")
        y_encoded, uniques = pd.factorize(y)
        if len(uniques) != 2:
            raise ValueError("Encoded target variable (y) must have exactly 2 unique values.")
        y = pd.Series(y_encoded, index=y.index)

        X = pd.DataFrame(X).copy()
        y = pd.Series(y).copy()
        cols = X.columns

        for feature in tqdm(cols, desc="Processing features"):
            logging.info(f"Processing feature: {feature}")
            if is_datetime64_any_dtype(X[feature]):
                logging.info(f"Skipping datetime feature: {feature}")
                continue

            d = self._bin_data(X[feature], y)
            d['% of Events'] = np.maximum(d['Events'], self.regularization) / d['Events'].sum()
            d['Non-Events'] = d['N'] - d['Events']
            d['% of Non-Events'] = np.maximum(d['Non-Events'], self.regularization) / d['Non-Events'].sum()
            d['WoE'] = np.log(d['% of Events'] / d['% of Non-Events']).replace([np.inf, -np.inf], 0)
            d['IV'] = (d['% of Events'] - d['% of Non-Events']) * d['WoE']

            if X[feature].nunique() == len(X[feature]):
                logging.info(f"Feature {feature} contains only unique values. Assigning WoE and IV value of 0.")
                d['WoE'] = 0
                d['IV'] = 0

            d.insert(loc=0, column='Feature', value=feature)

            # Logging the IV for each feature
            iv_value = d['IV'].sum()
            logging.info(f"Information value of {feature} is {iv_value:.6f}")

            temp = pd.DataFrame({"Feature": [feature], "IV": [iv_value]}, columns=["Feature", "IV"])
            self.iv = pd.concat([self.iv, temp], axis=0)
            self.woe = pd.concat([self.woe, d], axis=0)

        if self.retain_only_predictive_features:
            self.retain_predictive_features()

        if self.drop_invariant:
            self.drop_invariant_features()
        logging.info("Fit method completed.")
        return self

    def transform(self, X):
        """
        Transforms the data using the fitted WoE values.

        Parameters
        ----------
        X : Pandas DataFrame, shape [n_samples, n_features]
            The data to transform.

        Returns
        -------
        X_transformed : DataFrame
            The transformed data with encoded features.
        """
        check_is_fitted(self, 'woe')
        if self.woe.empty:
            raise ValueError("The WoE dataframe is empty. Please ensure the fit method has been called.")

        X_transformed = X.copy()[self.woe['Feature'].unique()]
        random_state_generator = check_random_state(self.random_state)

        for feature in tqdm(self.woe['Feature'].unique(), desc="Transforming features"):
            logging.info(f"Transforming feature: {feature}")
            woe_dict = self.woe[self.woe['Feature'] == feature][['Cutoff', 'WoE']].set_index('Cutoff')[
                'WoE'].to_dict()
            if is_datetime64_any_dtype(X_transformed[feature]):
                logging.warning(f"Skipping transformation for datetime variable: {feature}")
                continue

            if (X_transformed[feature].dtype.kind in 'bifc') and (len(np.unique(X[feature])) > 10):
                # For numerical variables, bin them as during fit
                X_transformed[f"{feature}"] = pd.qcut(X_transformed[feature], q=self.bins, duplicates='drop').map(
                    woe_dict)
                X_transformed[f"{feature}"] = X_transformed[f"{feature}"].astype('float64')

            else:
                # For categorical variables, directly map WoE values
                X_transformed[f"{feature}"] = X_transformed[feature].map(woe_dict).fillna(0)
                X_transformed[f"{feature}"] = X_transformed[f"{feature}"].astype('float64')
            if self.randomized:
                noise = random_state_generator.normal(0, self.sigma, X_transformed[feature].shape)
                X_transformed[feature] += noise

        logging.info("Transform method completed.")
        return X_transformed.to_numpy()

    def fit_transform(self, X, y=None, **kwargs):
        """
        Fits the encoder and transforms the dataset in one step.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data to fit and transform.
        y : array-like, shape [n_samples]
            The binary target variable.

        Returns
        -------
        X_transformed : DataFrame
            The transformed data with encoded features.
        """
        return self.fit(X, y).transform(X)

    def retain_predictive_features(self):
        """
        Eliminate features based on their predictive power.
        Keep features with IV values in the range of 0.02 to 0.5, indicating weak to strong predictive power.
        """
        # Check if the IV DataFrame is empty
        if self.iv.empty:
            logging.warning("IV DataFrame is empty. Please ensure the fit method has been called before filtering.")
            return

        original_feature_count = self.iv.shape[0]

        # Filter the IV DataFrame for features within the specified IV range
        filtered_iv = self.iv[(self.iv['IV'] >= 0.02) & (self.iv['IV'] <= 0.5)]

        # Update the self.iv with the filtered DataFrame
        self.iv = filtered_iv

        # Filter the self.woe DataFrame to keep only the features present in the filtered IV DataFrame
        self.woe = self.woe[self.woe['Feature'].isin(filtered_iv['Feature'])]

        dropped_features_count = original_feature_count - filtered_iv.shape[0]
        logging.info(f"Filtered {dropped_features_count} out of {original_feature_count} features based on IV values. "
                     f"Retained features with IV in the range 0.02 to 0.5.")

    def drop_invariant_features(self):
        """
            Drops features that are invariant after encoding.
        """
        invariant_features = self.woe.groupby('Feature')['WoE'].std().fillna(0) <= 0.0
        invariant_features = invariant_features[invariant_features].index.tolist()

        if invariant_features:
            # Drop invariant features from self.woe and self.iv
            self.woe = self.woe[~self.woe['Feature'].isin(invariant_features)].reset_index(drop=True)
            self.iv = self.iv[~self.iv['Feature'].isin(invariant_features)].reset_index(drop=True)
            logging.info(f"Dropped invariant features: {invariant_features}")
        else:
            logging.info("No invariant features found to drop.")

# if __name__ == '__main__':
#     sql_type = "mssql"
#     database = "AdventureWorks2022"
#     target_schema = "Sales"
#     target_table = "SalesOrderHeader"
#     target_attribute = "churn"
#     include_all_schemas = True
#     tables, primary_keys, fkg = get_data(sql_type=sql_type,
#                                          database=database,
#                                          target_schema=target_schema,
#                                          include_all_schemas=include_all_schemas)
#     tables = preprocess_tables(target_schema=target_schema, tables=tables)
#     df = tables['SalesOrderHeader'].drop(columns=['churn'])  # ['CreditCardApprovalCode']
#     y = tables['SalesOrderHeader']['churn']
#     woe = WoEEncoder(retain_only_predictive_features=True, drop_invariant=True)
#     scalar = woe.fit(df, y)
#     tmp = scalar.transform(df)
#     print(tmp)
