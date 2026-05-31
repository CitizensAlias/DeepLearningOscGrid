from catboost import CatBoostClassifier
import numpy as np

AllCat = CatBoostClassifier()
AllCat.load_model("CatBoost_AllCategories")  # predicts main categories: 0, 1, 2, 3

NormalCat = CatBoostClassifier()
NormalCat.load_model("CatBoost_0NoEvent")  # predicts subcategories: 0(NoEvent), 1(SomeEvent)

OperationalCat = CatBoostClassifier()
OperationalCat.load_model("CatBoost_1OperationalSwitching")  # predicts subcategories: 0(wrong category), 1, 11, 111, 12

AbnormalCat = CatBoostClassifier()
AbnormalCat.load_model("CatBoost_2AbnormalEvents")  # predicts subcategories: 0(wrong category), 2, 211, 212, 213, 22, 23, 231, 241, 242, 251, 26, 271, 272

FaultCat = CatBoostClassifier()
FaultCat.load_model("CatBoost_3FaultEvents")  # predicts subcategories: 0(wrong category), 3, 31, 32, 33, 34, 35


def five_cats_predict(entry):
    all_cat_prediction = AllCat.predict(entry, prediction_type='Probability')
    category = _get_category_with_highest_probability(all_cat_prediction)
    # Map index back to actual AllCat class label (0, 1, 2, 3)
    category = AllCat.classes_[category]

    match category:

        case 1:
            operational_cat_prediction = OperationalCat.predict(entry, prediction_type='Probability')
            operational_cat_subcategory = OperationalCat.classes_[
                _get_category_with_highest_probability(operational_cat_prediction)
            ]
            if operational_cat_subcategory == 0:
                return _go_through_every_cat(entry)
            return operational_cat_subcategory

        case 2:
            abnormal_cat_prediction = AbnormalCat.predict(entry, prediction_type='Probability')
            abnormal_cat_subcategory = AbnormalCat.classes_[
                _get_category_with_highest_probability(abnormal_cat_prediction)
            ]
            if abnormal_cat_subcategory == 0:
                return _go_through_every_cat(entry)
            return abnormal_cat_subcategory

        case 3:
            fault_cat_prediction = FaultCat.predict(entry, prediction_type='Probability')
            fault_cat_subcategory = FaultCat.classes_[
                _get_category_with_highest_probability(fault_cat_prediction)
            ]
            if fault_cat_subcategory == 0:
                return _go_through_every_cat(entry)
            return fault_cat_subcategory

        case _:  # case 0: expected NoEvent
            normal_cat_prediction = NormalCat.predict(entry, prediction_type='Probability')
            normal_cat_subcategory = NormalCat.classes_[
                _get_category_with_highest_probability(normal_cat_prediction)
            ]
            if normal_cat_subcategory != 0:  # NormalCat thinks something happened → escalate
                return _go_through_every_cat(entry)
            return 0


def _go_through_every_cat(entry):
    """
    Fallback: gather all valid (non-'wrong-category') class probabilities from every
    specialist model and return the class label with the globally highest probability.

    Class label 0 in NormalCat  = NoEvent         → keep as candidate "0"
    Class label 1 in NormalCat  = SomeOtherEvent  → discard (not a final category)
    Class label 0 in specialist models             → discard ("wrong category" sentinel)
    All other labels                               → keep as candidates
    """
    all_candidates = {}  # {final_category_label: probability}

    # --- NormalCat: keep only class 0 (NoEvent) ---
    normal_pred = NormalCat.predict(entry, prediction_type='Probability')[0]
    normal_labels = NormalCat.classes_
    for label, prob in zip(normal_labels, normal_pred):
        if label == 0:  # "NoEvent" is a valid final category
            all_candidates[label] = prob
        # label == 1 ("SomeEvent") is discarded — not a final output category

    # --- OperationalCat: skip sentinel class 0 ---
    op_pred = OperationalCat.predict(entry, prediction_type='Probability')[0]
    op_labels = OperationalCat.classes_
    for label, prob in zip(op_labels, op_pred):
        if label != 0:  # 0 means "wrong category", discard
            all_candidates[label] = prob

    # --- AbnormalCat: skip sentinel class 0 ---
    ab_pred = AbnormalCat.predict(entry, prediction_type='Probability')[0]
    ab_labels = AbnormalCat.classes_
    for label, prob in zip(ab_labels, ab_pred):
        if label != 0:
            all_candidates[label] = prob

    # --- FaultCat: skip sentinel class 0 ---
    fault_pred = FaultCat.predict(entry, prediction_type='Probability')[0]
    fault_labels = FaultCat.classes_
    for label, prob in zip(fault_labels, fault_pred):
        if label != 0:
            all_candidates[label] = prob

    # Return the label with the highest probability across all candidates
    most_probable_category = max(all_candidates, key=all_candidates.get)
    return most_probable_category

def _get_category_with_highest_probability(prediction):
    """
    CatBoost returns shape (1, n_classes) for a single entry with prediction_type='Probability'.
    Returns the class label with the highest probability.
    """
    probs = prediction[0]  # shape: (n_classes,)
    best_index = np.argmax(probs)
    return best_index  # index == class label only if classes are 0-indexed integers

