features = np.array(features)

result = permutation_importance(rfc, X_train_data, y_train, n_repeats=10,
								random_state=42)
perm_sorted_idx = result.importances_mean.argsort()

tree_importance_sorted_idx = np.argsort(rfc.feature_importances_)
print(tree_importance_sorted_idx)
tree_indices = np.arange(0, len(rfc.feature_importances_)) + 0.5

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
ax1.barh(tree_indices,
		 rfc.feature_importances_[tree_importance_sorted_idx], height=0.7)
ax1.set_yticklabels(features[tree_importance_sorted_idx])
ax1.set_yticks(tree_indices)
ax1.set_ylim((0, len(rfc.feature_importances_)))
ax2.boxplot(result.importances[perm_sorted_idx].T, vert=False,
			labels=features[perm_sorted_idx])
fig.tight_layout()
plt.show()