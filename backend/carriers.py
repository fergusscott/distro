"""
Simulated carrier appetite profiles. Each carrier has distinct preferences
that drive its own synthetic training data and ML model.

commission_rates: per-line broker commission as a decimal.
base_commission: fallback rate for lines not explicitly listed.
"""

CARRIERS = {
    "apex_specialty": {
        "id": "apex_specialty",
        "name": "Apex Specialty",
        "tagline": "Tech & Professional Lines",
        "base_commission": 0.13,
        "commission_rates": {
            "professional_liability": 0.14,
            "cyber": 0.15,
            "management_liability": 0.13,
            "general_liability": 0.12,
        },
        "appetite": {
            "preferred_lines": ["professional_liability", "cyber", "management_liability", "general_liability"],
            "avoid_lines": ["workers_comp", "commercial_auto"],
            "preferred_industries": ["technology", "financial_services", "healthcare"],
            "avoid_industries": ["construction", "manufacturing", "transportation", "hospitality"],
            "preferred_premium_range": (5000, 250000),
            "max_loss_ratio": 0.50,
            "high_risk_states": {"FL", "NY"},
            "min_years_in_business": 2,
        },
    },
    "meridian_mga": {
        "id": "meridian_mga",
        "name": "Meridian MGA",
        "tagline": "Contracting & Trades",
        "base_commission": 0.15,
        "commission_rates": {
            "general_liability": 0.15,
            "workers_comp": 0.10,
            "commercial_auto": 0.12,
            "commercial_property": 0.14,
        },
        "appetite": {
            "preferred_lines": ["general_liability", "workers_comp", "commercial_auto", "commercial_property"],
            "avoid_lines": ["cyber", "management_liability"],
            "preferred_industries": ["construction", "manufacturing", "transportation"],
            "avoid_industries": ["technology", "financial_services", "healthcare"],
            "preferred_premium_range": (8000, 300000),
            "max_loss_ratio": 0.65,
            "high_risk_states": {"CA", "NY"},
            "min_years_in_business": 3,
        },
    },
    "harbor_underwriters": {
        "id": "harbor_underwriters",
        "name": "Harbor Underwriters",
        "tagline": "Commercial Property & Casualty",
        "base_commission": 0.11,
        "commission_rates": {
            "commercial_property": 0.11,
            "general_liability": 0.12,
            "commercial_auto": 0.10,
        },
        "appetite": {
            "preferred_lines": ["commercial_property", "general_liability", "commercial_auto"],
            "avoid_lines": ["cyber", "management_liability", "professional_liability"],
            "preferred_industries": ["retail", "real_estate", "manufacturing", "hospitality"],
            "avoid_industries": ["technology", "financial_services"],
            "preferred_premium_range": (15000, 600000),
            "max_loss_ratio": 0.68,
            "high_risk_states": {"FL"},
            "min_years_in_business": 1,
        },
    },
    "summit_professional": {
        "id": "summit_professional",
        "name": "Summit Professional Lines",
        "tagline": "Management & Professional Liability",
        "base_commission": 0.14,
        "commission_rates": {
            "management_liability": 0.15,
            "professional_liability": 0.14,
            "cyber": 0.14,
        },
        "appetite": {
            "preferred_lines": ["management_liability", "professional_liability", "cyber"],
            "avoid_lines": ["workers_comp", "commercial_auto", "commercial_property"],
            "preferred_industries": ["financial_services", "technology", "healthcare", "real_estate"],
            "avoid_industries": ["construction", "transportation", "hospitality", "manufacturing"],
            "preferred_premium_range": (10000, 500000),
            "max_loss_ratio": 0.48,
            "high_risk_states": {"CA", "FL", "NY"},
            "min_years_in_business": 3,
        },
    },
}
