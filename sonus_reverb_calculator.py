import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# -----------------------------------------------
# 1. Define the Materials Library
# -----------------------------------------------
# Each entry has the format:
# 'Material Name': [absorption@160Hz, @250Hz, @500Hz, @800Hz, @1.250kHz, @2kHz, @3.150kHz]
MATERIALS_LIBRARY = {
    "Unpainted Concrete": [0.02, 0.02, 0.02, 0.03, 0.03, 0.04, 0.05],
    "Painted Concrete": [0.02, 0.03, 0.03, 0.03, 0.04, 0.05, 0.05],
    "Wood Flooring": [0.10, 0.07, 0.06, 0.05, 0.06, 0.07, 0.09],
    "Drywall": [0.15, 0.10, 0.05, 0.04, 0.04, 0.05, 0.05],
    "Glass Window": [0.02, 0.03, 0.03, 0.03, 0.02, 0.02, 0.02],
    "Carpet (on concrete)":   [0.04, 0.10, 0.20, 0.35, 0.50, 0.60, 0.65],
    "Open Plenum Ceiling": [0.05, 0.07, 0.08, 0.10, 0.12, 0.13, 0.15],
    "Acoustic Ceiling Tile": [0.35, 0.65, 0.75, 0.80, 0.85, 0.85, 0.90],
    "Unpainted Brick": [0.02, 0.03, 0.04, 0.05, 0.05, 0.05, 0.06],
    # Add more materials as needed...
}

TREATMENT_LIBRARY = {
    "1 Inch Fabric Panel":      [0.16, 0.30, 0.79, 0.97, 1.09, 1.13, 1.08],
    "2 Inch Fabric Panel":      [0.72, 0.93, 1.11, 1.07, 1.02, 1.02, 1.01],
    "Trava Acoustic Wood Panel": [0.50, 0.71, 0.94, 0.89, 0.86, 0.81, 0.81],
    "Aoudi Wood Plank System": [0.50, 0.71, 0.94, 0.89, 0.86, 0.81, 0.81],
    "Shape Shift Felt Wall Panels": [0.02, 0.05, 0.20, 0.53, 0.86, 1.24, 1.22],
    "Loda Acoustic Wood Baffles": [0.28, 0.79, 1.15, 1.27, 1.24, 1.31, 1.34],
    "Straight Away Felt Baffle": [0.23, 0.39, 0.59, 0.68, 0.85, 1.05, 1.18],
    "Serial Box Felt Baffle": [0.34, 0.52, 0.88, 1.06, 1.14, 1.31, 1.36],
    "Jimmy Beam Felt Baffle": [0.25, 0.38, 0.72, 0.91, 1.20, 1.36, 1.36],
    "Fabric Wrapped Baffle": [0.19, 0.63, 1.19, 1.63, 1.74, 1.69, 1.58],
    "Olino Cylindrical Baffle": [0.44, 1.13, 1.77, 1.94, 1.94, 1.84, 1.70],
    "Rib It Felt Wall Panel": [0.33, 0.30, 0.60, 1.03, 1.10, 1.20, 1.08,],
    "Grille Great (Wall)": [0.39, 0.60, 1.10, 1.18, 1.20, 1.30, 1.23,],
    "Grille Great (Ceiling)": [1.12, 1.10, 1.00, 0.97, 1.00, 1.00, 1.03],
    "Side Step" : [0.45, 0.78, 1.09, 1.18, 1.12, 1.06, 1.01,],
    "Rockin' Roll Felt Ceiling System" : [0.23, 0.39, 0.59, 0.68, 0.85, 1.05, 1.18],
    "Paradigm Shift" : [0.26, 0.13, 0.26, 0.56, 0.69, 0.81, 0.90,],
    "5th Dimension Felt Grid System": [0.39, 0.54, 0.63, 0.86, 0.98, 1.19, 1.30, 0.82],
    "Surround Stratus Felt Cloud": [0.67, 0.54, 1.01, 1.48, 1.79, 1.95, 1.84,],
    "Waffle Iron": [0.13, 0.28, 0.55, 0.73, 0.94, 1.33, 1.49],
    "Slat Attack": [0.21, 0.40, 0.70, 0.90, 1.10, 1.04],
    "Truss Me Baffle": [0.23, 0.39, 0.59, 0.68, 0.85, 1.05, 1.05, 1.18]

    # Add more treatments as needed...
}
ROOM_TYPE_IDEALS = {
    "Classroom": {
        160: (1.0, 1.2),
        250: (0.9, 1.2),
        500: (0.8, 1.0),
        800: (0.8, 1.0),
        1250: (0.8, 1.0),
        2000: (0.8, 1.0),
        3150: (0.8, 1.0)
    },
    "Conference Room": {
        160: (0.9, 1.2),
        250: (0.9, 1.2),
        500: (0.8, 1.0),
        800: (0.8, 1.0),
        1250: (0.7, 0.9),
        2000: (0.7, 0.9),
        3150: (0.7, 0.9)
    },
        "Open Office": {
        160: (1.0, 1.3),
        250: (1.0, 1.3),
        500: (0.9, 1.1),
        800: (0.8, 1.0),
        1250: (0.8, 1.0),
        2000: (0.8, 1.0),
        3150: (0.8, 1.0)
    },
        "Recording Studio": {
        160: (0.4, 0.7),
        250: (0.4, 0.6),
        500: (0.3, 0.5),
        800: (0.3, 0.5),
        1250: (0.3, 0.4),
        2000: (0.3, 0.4),
        3150: (0.3, 0.4)
    },
         "Waiting Room": {
        160: (1.1, 1.3),
        250: (1.0, 1.2),
        500: (0.9, 1.1),
        800: (0.8, 1.0),
        1250: (0.8, 1.0),
        2000: (0.8, 1.0),
        3150: (0.7, 1.0)
    },
        "Hotel Lobby": {
        160: (1.2, 1.6),
        250: (1.0, 1.4),
        500: (0.9, 1.2),
        800: (0.9, 1.2),
        1250: (0.8, 1.1),
        2000: (0.8, 1.2),
        3150: (0.8, 1.1)
    },
        "Gym": {
        160: (2.0, 2.3),
        250: (1.8, 2.2),
        500: (1.5, 2.0),
        800: (1.5, 2.0),
        1250: (1.5, 2.0),
        2000: (1.5, 2.0),
        3150: (1.5, 2.0)
    },
        "Restaurant": {
        160: (1.0, 1.4),
        250: (0.9, 1.3),
        500: (0.8, 1.2),
        800: (0.8, 1.0),
        1250: (0.8, 1.0),
        2000: (0.8, 1.0),
        3150: (0.8, 1.0)
    },
        "Coffee Shop": {
        160: (1.0, 1.3),
        250: (0.9, 1.2),
        500: (0.8, 1.1),
        800: (0.7, 1.0),
        1250: (0.7, 1.0),
        2000: (0.7, 1.0),
        3150: (0.7, 1.0)
    },

    # Add more room types as desired...


}



# Frequencies for which we have absorption data
FREQUENCIES = [160, 250, 500, 800, 1250, 2000, 3150]

# -----------------------------------------------
# 2. Helper Functions
# -----------------------------------------------
def calculate_room_stats(length_ft, width_ft, height_ft):
    """
    Given room dimensions, return:
    - Volume (ft^3)
    - Floor / Ceiling area (each, ft^2)
    - Total surface area (walls + floor + ceiling, ft^2)
      (not counting windows, doors, etc. for simplicity)
    """
    volume = length_ft * width_ft * height_ft
    floor_area = length_ft * width_ft
    ceiling_area = length_ft * width_ft  # same as floor
    # Perimeter walls:
    long_wall_area = length_ft * height_ft
    short_wall_area = width_ft * height_ft
    # Two long walls + two short walls:
    total_wall_area = 2 * long_wall_area + 2 * short_wall_area
    total_surface_area = floor_area + ceiling_area + total_wall_area

    return {
        "volume": volume,
        "floor_area": floor_area,
        "ceiling_area": ceiling_area,
        "long_wall_area": long_wall_area,
        "short_wall_area": short_wall_area,
        "total_surface_area": total_surface_area
    }

def calculate_sabine_rt60(volume, materials_dict):
    """
    Calculates the RT60 for each frequency band using Sabine's formula:
    RT60 = 0.161 * (Volume / Total_Absorption)
    (0.161 is the approximate factor if Volume is in m^3; for ft^3, often
    a factor ~0.049 is used. Here we assume SI units. 
    We will do a direct approach: first convert ft^3 -> m^3.)
    
    1 cubic foot = 0.0283168 cubic meters
    => volume_m3 = volume_ft3 * 0.0283168

    materials_dict: dictionary where
        key: (material_name)
        value: (area in ft^2, absorption_coefs [list])
    """
    # Convert volume to cubic meters
    volume_m3 = volume * 0.0283168
    
    # Prepare total absorption area (m^2) at each frequency
    # 1 ft^2 = 0.092903 m^2
    total_absorption = np.zeros(len(FREQUENCIES))

    for material_name, (area_ft2, coeffs) in materials_dict.items():
        # Convert area to m^2
        area_m2 = area_ft2 * 0.092903
        # Each frequency => area * absorption coefficient
        freq_absorption = np.array(coeffs) * area_m2
        total_absorption += freq_absorption

    # Calculate RT60 for each frequency using Sabine
    # RT60 = 0.161 * (Volume_m3 / Absorption_m2) 
    # (0.161 is a standard approximation in SI)
    # Protect against divide-by-zero if total_absorption is 0
    rt60 = np.where(total_absorption > 0.0,
                    0.161 * volume_m3 / total_absorption,
                    0.0)
    return rt60

def add_material_selector(library, index, key_prefix):
    """
    Creates a row in Streamlit for selecting a material from `library` 
    and the area in ft^2. Returns the chosen material name and area.
    """
    cols = st.columns([2, 1])
    with cols[0]:
        mat_name = st.selectbox(
            f"Material #{index+1} - Select Type",
            options=list(library.keys()),
            key=f"{key_prefix}_material_{index}"
        )
    with cols[1]:
        area = st.number_input(
            f"Area (ft²) for Material #{index+1}",
            min_value=0.0, value=0.0, step=1.0,
            key=f"{key_prefix}_area_{index}"
        )
    return mat_name, area

# -----------------------------------------------
# 3. Streamlit App Layout
# -----------------------------------------------
def main():
    st.title("Sonus Room Reverberation Calculator")
    st.write(
        "This application calculates the RT60 (reverb time) "
        "of a rectangular room using Sabine's Equation. "
        "Add Sonus products to see how they treat your space!"
    )

    # Room Type Selection
    st.header("Select a Room Type")
    room_type = st.selectbox(
        "Room Type",
        list(ROOM_TYPE_IDEALS.keys()),
        index=0
    )
    # Room dimension inputs
    st.header("Room Dimensions")
    col_dim1, col_dim2, col_dim3 = st.columns(3)
    with col_dim1:
        length_ft = st.number_input("Length (ft)", min_value=1.0, value=20.0)
    with col_dim2:
        width_ft = st.number_input("Width (ft)", min_value=1.0, value=15.0)
    with col_dim3:
        height_ft = st.number_input("Height (ft)", min_value=1.0, value=10.0)
    
    # Calculate initial room stats
    stats = calculate_room_stats(length_ft, width_ft, height_ft)
    
        # Placeholder data for the report
    ideal_ranges = ROOM_TYPE_IDEALS.get(room_type, {})
    min_ideal = min([r[0] for r in ideal_ranges.values()])
    max_ideal = max([r[1] for r in ideal_ranges.values()])

    data = {
        "[RoomType]": room_type,
        "[Ideal RT]": f"{min_ideal:.1f} - {max_ideal:.1f} seconds",
        "[Volume]": f"{stats['volume']:.2f} ft³",
        "[RT60 Untreated]": "1.5 seconds",  # Example placeholder
        "[RT60 Treated]": "0.8 seconds",  # Example placeholder
        "[Date]": "01/01/2025",  # Example placeholder
        "[Project Name]": "Conference Room Acoustic Analysis"
    }
    
    # Display stats in an info box
    st.subheader("Room Stats")
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    col_stat1.metric("Volume (ft³)", f"{stats['volume']:.2f}")
    col_stat2.metric("Floor (ft²)", f"{stats['floor_area']:.2f}")
    col_stat3.metric("Ceiling (ft²)", f"{stats['ceiling_area']:.2f}")
    col_stat4.metric("Wall Area (ft²)", f"{2*stats['long_wall_area'] + 2*stats['short_wall_area']:.2f}")

    total_room_area = stats["total_surface_area"]
    
    st.write("---")
    
    # -------------------------------------------
    # Materials Section - Untreated Surfaces
    # -------------------------------------------
    st.header("Room Materials")
    st.write(
        "Specify existing surfaces and materials in the room before treatment. "
        "You can add multiple lines to represent each surface material."
    )
    
    # Dynamic input for materials
    st.subheader("Untreated Materials")
    if "material_count" not in st.session_state:
        st.session_state["material_count"] = 1  # start with one row
    
    # Buttons to add/remove rows
    col_btn_add, col_btn_remove = st.columns(2)
    with col_btn_add:
        if st.button("Add Another Material"):
            st.session_state["material_count"] += 1
    with col_btn_remove:
        if st.button("Remove Last Material"):
            if st.session_state["material_count"] > 1:
                st.session_state["material_count"] -= 1

    # Collect user input for each material row
    material_entries = {}
    total_material_area = 0.0
    for i in range(st.session_state["material_count"]):
        mat_name, area = add_material_selector(MATERIALS_LIBRARY, i, "untreated")
        material_entries[f"untreated_mat_{i}"] = (mat_name, area)
        total_material_area += area
    
    # Remaining area
    remaining_area = total_room_area - total_material_area
    st.info(f"**Remaining area in the room (ft²):** {remaining_area:.2f}")

    # -------------------------------------------
    # Materials Section - Treatments
    # -------------------------------------------
    st.header("Acoustic Treatments")
    st.write(
        "Specify which Sonus products to add to the room. "
        "These absorption values will be added to the untreated surfaces."
    )
    
    if "treatment_count" not in st.session_state:
        st.session_state["treatment_count"] = 1  # start with one row

    col_treat_add, col_treat_remove = st.columns(2)
    with col_treat_add:
        if st.button("Add Treatment Material"):
            st.session_state["treatment_count"] += 1
    with col_treat_remove:
        if st.button("Remove Treatment Material"):
            if st.session_state["treatment_count"] > 1:
                st.session_state["treatment_count"] -= 1

    treatment_entries = {}
    total_treatment_area = 0.0
    for i in range(st.session_state["treatment_count"]):
        mat_name, area = add_material_selector(TREATMENT_LIBRARY, i, "treated")
        treatment_entries[f"treated_mat_{i}"] = (mat_name, area)
        total_treatment_area += area

        # Room Type Selection
    room_type = "Classroom"  # Example, replace with actual selection logic
    length_ft = 20.0  # Replace with actual input logic
    width_ft = 15.0   # Replace with actual input logic
    height_ft = 10.0  # Replace with actual input logic
    stats = calculate_room_stats(length_ft, width_ft, height_ft)

    # Untreated Materials
    untreated_dict = {}
    for key, val in material_entries.items():
        mat_name, area_ft2 = val
        if area_ft2 > 0:
            untreated_dict[mat_name] = (
                untreated_dict.get(mat_name, (0, MATERIALS_LIBRARY[mat_name]))[0] + area_ft2,
                MATERIALS_LIBRARY[mat_name]
            )

    # Treated Materials
    treated_dict = untreated_dict.copy()
    for key, val in treatment_entries.items():
        mat_name, area_ft2 = val
        if area_ft2 > 0:
            if mat_name in treated_dict:
                existing_area, existing_coeffs = treated_dict[mat_name]
                treated_dict[mat_name] = (existing_area + area_ft2, existing_coeffs)
            else:
                treated_dict[mat_name] = (area_ft2, TREATMENT_LIBRARY[mat_name])

    # Calculate RT60
    rt60_untreated = calculate_sabine_rt60(stats["volume"], untreated_dict)
    rt60_treated = calculate_sabine_rt60(stats["volume"], treated_dict)

        # Ideal RT Range Calculation
    ideal_ranges = ROOM_TYPE_IDEALS.get(room_type, {})
    min_ideal = min([r[0] for r in ideal_ranges.values()])
    max_ideal = max([r[1] for r in ideal_ranges.values()])
    ideal_rt_range = f"{min_ideal:.1f} - {max_ideal:.1f}"


    st.write("---")

    # -------------------------------------------
    # Calculate RT60 (Untreated vs Treated)
    # -------------------------------------------
    # Build dict needed for the Sabine function: {material_name: (area, [coeffs])}
    untreated_dict = {}
    for key, val in material_entries.items():
        mat_name, area_ft2 = val
        if area_ft2 > 0:
            untreated_dict[mat_name] = (
                untreated_dict.get(mat_name, (0, MATERIALS_LIBRARY[mat_name]))[0] + area_ft2,
                MATERIALS_LIBRARY[mat_name]
            )

    # For "treated", combine the untreated surfaces + added treatments
    treated_dict = {}
    # Start with untreated
    for mat_name, (area_ft2, coeffs) in untreated_dict.items():
        treated_dict[mat_name] = (area_ft2, coeffs)

    # Add in treatments
    for key, val in treatment_entries.items():
        mat_name, area_ft2 = val
        if area_ft2 > 0:
            if mat_name in treated_dict:
                existing_area, existing_coeffs = treated_dict[mat_name]
                treated_dict[mat_name] = (existing_area + area_ft2, existing_coeffs)
            else:
                treated_dict[mat_name] = (area_ft2, TREATMENT_LIBRARY[mat_name])

    # Now calculate RT60 for each
    rt60_untreated = calculate_sabine_rt60(stats["volume"], untreated_dict)
    rt60_treated = calculate_sabine_rt60(stats["volume"], treated_dict)

    # -------------------------------------------
    # Plot Results
    # -------------------------------------------
    st.header("Calculated Reverberation Times (RT60)")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(FREQUENCIES, rt60_untreated, marker='o', label="Untreated")
    ax.plot(FREQUENCIES, rt60_treated, marker='o', label="Treated")
    ax.set_xscale('log')
    ax.set_xticks(FREQUENCIES)
    ax.set_xticklabels(FREQUENCIES)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("RT60 (seconds)")
    ax.set_title("RT60 vs. Frequency")
    ax.legend()
    ax.grid(True, which="both", linestyle="--", alpha=0.7)


    
        # -- Add the Shaded Area for the “Ideal” range
    # For each frequency band in FREQUENCIES, get the min and max from the “ideal”
    # and shade that region. We'll approximate “band edges” by taking the midpoint
    # between freq[i] and freq[i+1].
    
    for i, freq in enumerate(FREQUENCIES):
        if i < len(FREQUENCIES) - 1:
            freq_left  = freq
            freq_right = FREQUENCIES[i+1]
            # We'll define a band from freq_left to freq_right
        else:
            # For the last frequency, we'll just pick a small band or replicate
            freq_left  = freq
            freq_right = freq * 1.2  # e.g., 20% higher just for illustration

        (ideal_min, ideal_max) = ROOM_TYPE_IDEALS[room_type].get(freq, (0, 0))
        
        # Actually fill the region:
        ax.fill_between(
            [freq_left, freq_right],  # x range
            [ideal_min, ideal_min],   # lower boundary
            [ideal_max, ideal_max],   # upper boundary
            color='green', alpha=0.3  # light green shading
        )
    
    st.pyplot(fig)

    # Display numeric RT60 values in a table
    results_df = pd.DataFrame({
        "Frequency (Hz)": FREQUENCIES,
        "RT60 - Untreated (s)": rt60_untreated,
        "RT60 - Treated (s)": rt60_treated
    })
    st.table(results_df.round(2))

    # -------------------------------------------
# Generate Report Function
# -------------------------------------------


if __name__ == "__main__":
    main()