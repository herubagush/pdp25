"""
ST-DBSCAN Module (FIXED & OPTIMIZED)
Perbaikan: efficiency, code duplication, memory usage
"""

import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
from sklearn.metrics import silhouette_score, davies_bouldin_score


# ============= UTILITY FUNCTIONS =============

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Fungsi Haversine terpusat untuk menghindari duplikasi
    Mendukung vectorized operations untuk efisiensi
    """
    R = 6371.0
    
    # Convert to numpy arrays jika belum
    lat1, lon1, lat2, lon2 = map(np.asarray, [lat1, lon1, lat2, lon2])
    
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    delta_lat = np.radians(lat2 - lat1)
    delta_lon = np.radians(lon2 - lon1)
    
    a = np.sin(delta_lat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    return R * c


def calculate_max_spatial_spread_efficient(lats, lons):
    """
    Hitung max spatial spread dengan lebih efisien
    Menggunakan vectorized operations
    """
    if len(lats) < 2:
        return 0
    
    # Untuk dataset kecil (<100), gunakan full pairwise
    if len(lats) < 100:
        n = len(lats)
        max_dist = 0
        for i in range(n):
            for j in range(i+1, n):
                dist = haversine_distance(lats[i], lons[i], lats[j], lons[j])
                max_dist = max(max_dist, dist)
        return max_dist
    
    # Untuk dataset besar, gunakan sampling strategis
    # Hitung jarak dari setiap titik ke centroid
    centroid_lat = np.mean(lats)
    centroid_lon = np.mean(lons)
    
    distances_to_centroid = haversine_distance(
        lats, lons, 
        centroid_lat, centroid_lon
    )
    
    # Ambil 2 titik terjauh dari centroid
    idx_sorted = np.argsort(distances_to_centroid)
    idx1, idx2 = idx_sorted[-1], idx_sorted[-2]
    
    # Hitung jarak antara 2 titik terjauh
    max_dist = haversine_distance(lats[idx1], lons[idx1], lats[idx2], lons[idx2])
    
    return max_dist


def convert_times_to_days(times):
    """Konversi waktu ke hari (fungsi terpusat)"""
    if isinstance(times[0], (datetime, pd.Timestamp, np.datetime64)):
        times_pd = pd.to_datetime(times)
        base_time = times_pd.min()
        return np.array([(t - base_time).total_seconds() / 86400 for t in times_pd])
    return np.array(times, dtype=float)


def clear_progress_widgets(progress_bar, status_text):
    """Bersihkan progress bar dan status text"""
    progress_bar.empty()
    status_text.empty()


# ============= STDBSCAN CLASS =============

class STDBSCAN:
    """
    Spatio-Temporal DBSCAN (OPTIMIZED)
    """
    
    def __init__(self, eps1=50, eps2=30, eps3=None, min_samples=5, use_depth=False):
        self.eps1 = eps1
        self.eps2 = eps2
        self.eps3 = eps3
        self.min_samples = min_samples
        self.use_depth = use_depth
        self.labels_ = None
        
    def _get_neighbors(self, point_idx, spatial_coords, temporal_coords, depth_coords=None):
        """Mencari tetangga (OPTIMIZED dengan vectorization)"""
        # Vectorized distance calculation
        dist_spatial = haversine_distance(
            spatial_coords[point_idx, 0], 
            spatial_coords[point_idx, 1],
            spatial_coords[:, 0], 
            spatial_coords[:, 1]
        )
        
        dist_temporal = np.abs(temporal_coords - temporal_coords[point_idx])
        
        # Boolean mask untuk spatial dan temporal
        mask = (dist_spatial <= self.eps1) & (dist_temporal <= self.eps2)
        
        # Tambahkan depth constraint jika perlu
        if self.use_depth and depth_coords is not None and self.eps3 is not None:
            dist_depth = np.abs(depth_coords - depth_coords[point_idx])
            mask = mask & (dist_depth <= self.eps3)
        
        # Exclude point itu sendiri
        mask[point_idx] = False
        
        return np.where(mask)[0].tolist()
    
    def fit(self, latitudes, longitudes, times, depths=None, debug=False):
        """Melakukan klasterisasi ST-DBSCAN"""
        n_points = len(latitudes)
        spatial_coords = np.column_stack([latitudes, longitudes])
        temporal_coords = convert_times_to_days(times)
        
        depth_coords = None
        if self.use_depth:
            if depths is None:
                raise ValueError("Parameter 'depths' harus diisi jika use_depth=True")
            if self.eps3 is None:
                raise ValueError("Parameter 'eps3' harus diisi jika use_depth=True")
            depth_coords = np.array(depths, dtype=float)
        
        if debug:
            st.write(f"\n**Debug - Konversi Data:**")
            st.write(f"- Mode: {'4D (Spasial-Temporal-Depth)' if self.use_depth else '3D (Spasial-Temporal)'}")
            st.write(f"- Range waktu: {temporal_coords.min():.2f} - {temporal_coords.max():.2f} hari")
            if self.use_depth and depth_coords is not None:
                st.write(f"- Range kedalaman: {depth_coords.min():.1f} - {depth_coords.max():.1f} km")
        
        # Initialize labels
        labels = np.full(n_points, None, dtype=object)
        cluster_id = 0
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            for point_idx in range(n_points):
                # Update progress setiap 5%
                if point_idx % max(1, n_points // 20) == 0:
                    progress_bar.progress(point_idx / n_points)
                    status_text.text(f"Processing: {point_idx}/{n_points} points...")
                
                if labels[point_idx] is not None:
                    continue
                
                neighbors = self._get_neighbors(point_idx, spatial_coords, temporal_coords, depth_coords)
                
                if len(neighbors) < self.min_samples - 1:
                    labels[point_idx] = -1
                    continue
                
                # Core point - expand cluster
                labels[point_idx] = cluster_id
                seed_set = list(neighbors)
                i = 0
                
                while i < len(seed_set):
                    q_idx = seed_set[i]
                    
                    if labels[q_idx] == -1:
                        labels[q_idx] = cluster_id
                    
                    if labels[q_idx] is None:
                        labels[q_idx] = cluster_id
                        q_neighbors = self._get_neighbors(q_idx, spatial_coords, temporal_coords, depth_coords)
                        
                        if len(q_neighbors) >= self.min_samples - 1:
                            seed_set.extend([n for n in q_neighbors if n not in seed_set])
                    
                    i += 1
                
                cluster_id += 1
            
            # Convert None to -1
            labels = np.array([label if label is not None else -1 for label in labels])
            
            progress_bar.progress(1.0)
            mode_text = "4D (Spasial-Temporal-Depth)" if self.use_depth else "3D (Spasial-Temporal)"
            status_text.text(f"✅ Clustering completed ({mode_text})! Found {cluster_id} clusters")
            
        finally:
            # Clear progress widgets setelah 2 detik
            import time
            time.sleep(2)
            clear_progress_widgets(progress_bar, status_text)
        
        self.labels_ = labels
        return labels


# ============= CLUSTERING METRICS =============

def calculate_clustering_metrics(df, labels, eps1, eps2):
    """Menghitung metrik evaluasi clustering (OPTIMIZED)"""
    mask = labels != -1
    if mask.sum() < 2:
        return None
    
    df_clustered = df[mask].copy()
    labels_clustered = labels[mask]
    
    n_clusters = len(np.unique(labels_clustered))
    if n_clusters < 2:
        return None
    
    # Prepare features
    lats = df_clustered['latitude'].values
    lons = df_clustered['longitude'].values
    times = df_clustered['datetime'].values
    
    times_days = convert_times_to_days(times)
    
    # Convert to km
    mean_lat = np.mean(lats)
    lats_km = lats * 111.0
    lons_km = lons * 111.0 * np.cos(np.radians(mean_lat))
    
    # Normalize
    times_normalized = times_days / eps2 if eps2 > 0 else times_days
    lats_normalized = lats_km / eps1 if eps1 > 0 else lats_km
    lons_normalized = lons_km / eps1 if eps1 > 0 else lons_km
    
    X = np.column_stack([lats_normalized, lons_normalized, times_normalized])
    
    # Calculate metrics
    try:
        silhouette = silhouette_score(X, labels_clustered, metric='euclidean')
    except:
        silhouette = None
    
    try:
        davies_bouldin = davies_bouldin_score(X, labels_clustered)
    except:
        davies_bouldin = None
    
    # Inertia
    inertia = 0
    for cluster_id in np.unique(labels_clustered):
        cluster_mask = labels_clustered == cluster_id
        cluster_points = X[cluster_mask]
        
        if len(cluster_points) > 0:
            centroid = cluster_points.mean(axis=0)
            distances_sq = np.sum((cluster_points - centroid) ** 2, axis=1)
            inertia += distances_sq.sum()
    
    # Stats
    n_noise = (labels == -1).sum()
    n_clustered = mask.sum()
    noise_ratio = n_noise / len(labels) if len(labels) > 0 else 0
    
    cluster_sizes = [np.sum(labels_clustered == cid) for cid in np.unique(labels_clustered)]
    avg_cluster_size = np.mean(cluster_sizes) if cluster_sizes else 0
    
    return {
        'silhouette_score': silhouette,
        'davies_bouldin_index': davies_bouldin,
        'inertia': inertia,
        'n_clusters': n_clusters,
        'n_clustered_points': n_clustered,
        'n_noise': n_noise,
        'noise_ratio': noise_ratio,
        'avg_cluster_size': avg_cluster_size
    }


# ============= PATTERN CLASSIFICATION =============

def classify_cluster_pattern(cluster_data, eps2):
    """Klasifikasi pola cluster (OPTIMIZED)"""
    durasi = (cluster_data['datetime'].max() - cluster_data['datetime'].min()).days
    
    lats = cluster_data['latitude'].values
    lons = cluster_data['longitude'].values
    
    # Efficient spatial spread calculation
    max_spatial_spread = calculate_max_spatial_spread_efficient(lats, lons)
    
    # Mean spatial spread (approximate dengan sampling untuk dataset besar)
    if len(lats) < 50:
        # Full calculation untuk dataset kecil
        spatial_distances = []
        for i in range(len(lats)):
            for j in range(i+1, len(lats)):
                dist = haversine_distance(lats[i], lons[i], lats[j], lons[j])
                spatial_distances.append(dist)
        mean_spatial_spread = np.mean(spatial_distances) if spatial_distances else 0
    else:
        # Approximate dengan sampling
        sample_size = min(50, len(lats))
        sample_idx = np.random.choice(len(lats), sample_size, replace=False)
        sample_lats = lats[sample_idx]
        sample_lons = lons[sample_idx]
        
        spatial_distances = []
        for i in range(len(sample_lats)):
            for j in range(i+1, len(sample_lats)):
                dist = haversine_distance(sample_lats[i], sample_lons[i], 
                                         sample_lats[j], sample_lons[j])
                spatial_distances.append(dist)
        mean_spatial_spread = np.mean(spatial_distances) if spatial_distances else 0
    
    # Time gaps
    times_sorted = cluster_data['datetime'].sort_values()
    time_gaps = [(times_sorted.iloc[i+1] - times_sorted.iloc[i]).days 
                 for i in range(len(times_sorted)-1)]
    
    max_time_gap = max(time_gaps) if time_gaps else 0
    mean_time_gap = np.mean(time_gaps) if time_gaps else 0
    
    # Movement calculation
    cluster_sorted = cluster_data.sort_values('datetime')
    n = len(cluster_sorted)
    
    total_movement = 0
    if n >= 3:
        seg_size = n // 3
        seg1 = cluster_sorted.iloc[:seg_size]
        seg2 = cluster_sorted.iloc[seg_size:2*seg_size]
        seg3 = cluster_sorted.iloc[2*seg_size:]
        
        cent1 = (seg1['latitude'].mean(), seg1['longitude'].mean())
        cent2 = (seg2['latitude'].mean(), seg2['longitude'].mean())
        cent3 = (seg3['latitude'].mean(), seg3['longitude'].mean())
        
        movement_12 = haversine_distance(cent1[0], cent1[1], cent2[0], cent2[1])
        movement_23 = haversine_distance(cent2[0], cent2[1], cent3[0], cent3[1])
        total_movement = movement_12 + movement_23
    
    # Temporal intensity
    temporal_intensity = len(cluster_data) / max(durasi, 1)
    
    # Classification logic
    pattern = "UNDEFINED"
    confidence = 0
    description = ""
    
    if durasi < eps2:
        pattern = "OCCASIONAL"
        confidence = 90
        description = f"Burst gempa dalam waktu singkat ({durasi} hari)"
    elif durasi > 2 * eps2 and max_spatial_spread < 50 and max_time_gap < eps2:
        pattern = "STATIONARY"
        confidence = 85
        description = f"Aktivitas kontinu di lokasi tetap"
    elif max_time_gap > eps2 and mean_spatial_spread < 40:
        pattern = "REAPPEARING"
        confidence = 80
        description = f"Aktivitas berulang dengan gap {max_time_gap} hari"
    elif max_spatial_spread > 30 and durasi > eps2:
        pattern = "TRACK"
        confidence = 75
        description = f"Pergerakan spatial {max_spatial_spread:.1f} km"
    else:
        pattern = "STATIONARY" if durasi >= 2 * eps2 else "OCCASIONAL"
        confidence = 60
        description = f"Durasi {durasi} hari"
    
    return {
        'pattern': pattern,
        'confidence': confidence,
        'description': description,
        'durasi': durasi,
        'max_spatial_spread': max_spatial_spread,
        'mean_spatial_spread': mean_spatial_spread,
        'max_time_gap': max_time_gap,
        'mean_time_gap': mean_time_gap,
        'total_movement': total_movement,
        'temporal_intensity': temporal_intensity,
        'n_events': len(cluster_data)
    }


def analyze_clusters(df, labels, eps2):
    """Analisis statistik per cluster"""
    df_analysis = df.copy()
    df_analysis['cluster'] = labels
    
    n_clusters = len(np.unique(labels[labels >= 0]))
    n_noise = np.sum(labels == -1)
    
    stats = []
    patterns = []
    
    for cluster_id in sorted(np.unique(labels)):
        if cluster_id == -1:
            continue
        
        cluster_data = df_analysis[df_analysis['cluster'] == cluster_id]
        pattern_info = classify_cluster_pattern(cluster_data, eps2)
        
        stats.append({
            'Cluster': cluster_id,
            'Jumlah Events': len(cluster_data),
            'Mag Min': cluster_data['magnitude'].min(),
            'Mag Max': cluster_data['magnitude'].max(),
            'Mag Mean': cluster_data['magnitude'].mean(),
            'Depth Min (km)': cluster_data['depth'].min(),
            'Depth Max (km)': cluster_data['depth'].max(),
            'Durasi (hari)': pattern_info['durasi'],
            'Pola': pattern_info['pattern'],
            'Confidence (%)': pattern_info['confidence']
        })
        
        patterns.append({
            'cluster_id': cluster_id,
            'pattern_info': pattern_info
        })
    
    return pd.DataFrame(stats), n_clusters, n_noise, patterns


# ============= SAMPLE DATA GENERATOR =============

def generate_sample_data():
    """Generate contoh data gempabumi"""
    np.random.seed(42)
    
    clusters = [
        # Cluster 1: Jawa Timur
        {'n': 50, 'lat': -7.9, 'lon': 112.6, 'time_range': (10, 30), 
         'mag_range': (3.5, 5.0), 'depth_range': (10, 50)},
        # Cluster 2: Sumatra Barat
        {'n': 45, 'lat': -0.95, 'lon': 100.35, 'time_range': (50, 70),
         'mag_range': (4.0, 5.5), 'depth_range': (20, 80)},
        # Cluster 3: Bali
        {'n': 25, 'lat': -8.65, 'lon': 115.2, 'time_range': (85, 100),
         'mag_range': (3.8, 5.2), 'depth_range': (15, 60)},
    ]
    
    all_data = []
    
    for c in clusters:
        lat = c['lat'] + np.random.normal(0, 0.25, c['n'])
        lon = c['lon'] + np.random.normal(0, 0.25, c['n'])
        time = np.random.uniform(c['time_range'][0], c['time_range'][1], c['n'])
        mag = np.random.uniform(c['mag_range'][0], c['mag_range'][1], c['n'])
        depth = np.random.uniform(c['depth_range'][0], c['depth_range'][1], c['n'])
        
        all_data.append((lat, lon, time, mag, depth))
    
    # Add noise
    n_noise = 15
    lat_noise = np.random.uniform(-10, 2, n_noise)
    lon_noise = np.random.uniform(95, 120, n_noise)
    time_noise = np.random.uniform(0, 100, n_noise)
    mag_noise = np.random.uniform(3.5, 4.5, n_noise)
    depth_noise = np.random.uniform(5, 100, n_noise)
    all_data.append((lat_noise, lon_noise, time_noise, mag_noise, depth_noise))
    
    # Concatenate
    latitudes = np.concatenate([d[0] for d in all_data])
    longitudes = np.concatenate([d[1] for d in all_data])
    times = np.concatenate([d[2] for d in all_data])
    magnitudes = np.concatenate([d[3] for d in all_data])
    depths = np.concatenate([d[4] for d in all_data])
    
    start_date = pd.Timestamp('2024-01-01')
    datetimes = [start_date + pd.Timedelta(days=float(t)) for t in times]
    
    df = pd.DataFrame({
        'datetime': datetimes,
        'latitude': latitudes,
        'longitude': longitudes,
        'magnitude': magnitudes,
        'depth': depths
    })
    
    return df.sort_values('datetime').reset_index(drop=True)


# ============= K-NN DISTANCE ANALYSIS =============

def calculate_knn_distances(latitudes, longitudes, k=5):
    """
    Menghitung jarak k-nearest neighbors (OPTIMIZED)
    Menggunakan vectorized operations untuk speedup
    """
    n_points = len(latitudes)
    knn_distances = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        for i in range(n_points):
            if i % max(1, n_points // 20) == 0:
                progress_bar.progress(i / n_points)
                status_text.text(f"Calculating k-NN distances: {i}/{n_points}...")
            
            # Vectorized distance calculation
            distances = haversine_distance(
                latitudes[i], longitudes[i],
                latitudes, longitudes
            )
            
            # Remove self-distance (0)
            distances = distances[distances > 0]
            
            # Sort dan ambil k terdekat
            distances_sorted = np.sort(distances)
            if len(distances_sorted) >= k:
                knn_distances.append(distances_sorted[k-1])
            else:
                knn_distances.append(distances_sorted[-1] if len(distances_sorted) > 0 else 0)
        
        progress_bar.progress(1.0)
        status_text.text("✅ k-NN distance calculation completed!")
        
    finally:
        import time
        time.sleep(2)
        clear_progress_widgets(progress_bar, status_text)
    
    return np.sort(knn_distances)


def find_knee_point(distances):
    """Mencari knee point dengan maximum curvature"""
    n = len(distances)
    
    # Gradient-based curvature
    dy = np.gradient(distances)
    d2y = np.gradient(dy)
    
    curvature = np.abs(d2y) / (1 + dy**2)**1.5
    
    # Cari maximum curvature (skip 10% awal dan akhir)
    start_idx = int(0.1 * n)
    end_idx = int(0.9 * n)
    
    knee_idx = start_idx + np.argmax(curvature[start_idx:end_idx])
    knee_distance = distances[knee_idx]
    
    return knee_idx, knee_distance


def find_optimal_eps1(df, k_range=[3, 5, 7], percentile_range=[90, 95, 99]):
    """Mencari eps1 optimal menggunakan k-distance plot"""
    latitudes = df['latitude'].values
    longitudes = df['longitude'].values
    
    results = {
        'k_results': {},
        'percentile_results': {},
        'recommendation': None
    }
    
    st.info("🔍 Menganalisis k-distance untuk berbagai nilai k...")
    
    for k in k_range:
        st.write(f"**Menghitung untuk k={k}...**")
        
        sorted_distances = calculate_knn_distances(latitudes, longitudes, k=k)
        knee_idx, knee_eps1 = find_knee_point(sorted_distances)
        
        results['k_results'][k] = {
            'sorted_distances': sorted_distances,
            'knee_index': knee_idx,
            'knee_eps1': knee_eps1
        }
        
        st.write(f"  - Knee point eps1 untuk k={k}: **{knee_eps1:.2f} km**")
    
    # Percentile analysis
    st.write("\n**Analisis Persentil:**")
    k_middle = k_range[len(k_range)//2]
    distances = results['k_results'][k_middle]['sorted_distances']
    
    for percentile in percentile_range:
        eps1_percentile = np.percentile(distances, percentile)
        results['percentile_results'][percentile] = eps1_percentile
        st.write(f"  - Persentil {percentile}: **{eps1_percentile:.2f} km**")
    
    # Recommendation
    knee_values = [results['k_results'][k]['knee_eps1'] for k in k_range]
    recommended_eps1 = np.median(knee_values)
    results['recommendation'] = recommended_eps1
    
    st.success(f"\n💡 **Rekomendasi eps1:** {recommended_eps1:.2f} km (median dari knee points)")
    
    return results


def plot_knn_distance(optimal_eps1_results, selected_k=None):
    """Plot k-distance untuk visualisasi"""
    import plotly.graph_objects as go
    
    k_results = optimal_eps1_results['k_results']
    
    if selected_k is None:
        selected_k = list(k_results.keys())[0]
    
    if selected_k not in k_results:
        selected_k = list(k_results.keys())[0]
    
    result = k_results[selected_k]
    distances = result['sorted_distances']
    knee_idx = result['knee_index']
    knee_eps1 = result['knee_eps1']
    
    fig = go.Figure()
    
    # Main curve
    fig.add_trace(go.Scatter(
        x=list(range(len(distances))),
        y=distances,
        mode='lines',
        name=f'{selected_k}-distance',
        line=dict(color='blue', width=2)
    ))
    
    # Knee point
    fig.add_trace(go.Scatter(
        x=[knee_idx],
        y=[knee_eps1],
        mode='markers',
        name='Knee Point',
        marker=dict(color='red', size=12, symbol='star'),
        hovertemplate=f'Knee Point<br>Index: {knee_idx}<br>eps1: {knee_eps1:.2f} km<extra></extra>'
    ))
    
    # Horizontal line at knee
    fig.add_hline(
        y=knee_eps1,
        line_dash="dash",
        line_color="red",
        annotation_text=f"eps1 = {knee_eps1:.2f} km",
        annotation_position="right"
    )
    
    # Percentile lines
    percentile_results = optimal_eps1_results['percentile_results']
    colors = ['green', 'orange', 'purple']
    for i, (percentile, value) in enumerate(percentile_results.items()):
        fig.add_hline(
            y=value,
            line_dash="dot",
            line_color=colors[i % len(colors)],
            annotation_text=f"P{percentile}: {value:.2f} km",
            annotation_position="left"
        )
    
    fig.update_layout(
        title=f"k-Distance Plot (k={selected_k}) untuk Menentukan eps1 Optimal",
        xaxis_title="Points (sorted by distance)",
        yaxis_title="k-th Nearest Neighbor Distance (km)",
        height=500,
        hovermode='closest',
        showlegend=True
    )
    
    return fig
